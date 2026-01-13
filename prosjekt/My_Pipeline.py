from pathlib import Path
import pandas as pd

Start_date = pd.Timestamp('2024-09-01 00:00:00+00:00')

Target_stations = [
    'Møllendalsplass', 'Torgallmenningen', 'Grieghallen', 'Høyteknologisenteret',
    'Studentboligene', 'Akvariet', 'Damsgårdsveien 71', 'Dreggsallmenningen Sør',
    'Florida Bybanestopp'
]

def workspace_paths(raw_data):
    here = Path(__file__).resolve().parent
    raw = here / raw_data
    return raw / 'stations.csv', raw / 'trips.csv', raw / 'weather.csv', here

def coerce_ts(s):
    # Parse til UTC timezone og rund ned til nærmeste hele time (unngå framtidslekkasje)
    return pd.to_datetime(s, utc=True, errors='coerce').dt.floor('h')


def load_weather(weather_fp, end_date):
    w = pd.read_csv(weather_fp, usecols=['timestamp', 'temperature', 'precipitation', 'wind_speed'])
    w['timestamp'] = coerce_ts(w['timestamp'])
    w = w[(w['timestamp'] >= Start_date) & (w['timestamp'] <= end_date)].copy()
    # Aggreger til time-nivå (hvis duplikater: mean temp/wind, sum precip)
    w = w.groupby('timestamp', as_index=False).agg({'temperature': 'mean', 'precipitation': 'sum', 'wind_speed': 'mean'})
    # Lager fullstendig time-grid og fyll NaN med kun ffill (ikke bfill for å unngå data leakage)
    full_hours = pd.date_range(Start_date.floor('h'), end_date.floor('h'), freq='h', tz='UTC')
    w = w.set_index('timestamp').reindex(full_hours).ffill().reset_index().rename(columns={'index': 'timestamp'})
    # Fyll evt. gjenværende NaN (helt i starten) med kolonnegjennomsnitt
    w = w.fillna(w.mean(numeric_only=True))
    return w


def load_stations(stations_fp):
    s = pd.read_csv(stations_fp)
    s['timestamp'] = coerce_ts(s['timestamp'])
    end_date = s['timestamp'].max()  # Dynamisk slutt-dato fra faktisk data
    s = s[(s['timestamp'] >= Start_date) & (s['timestamp'] <= end_date)].copy()
    s = s.drop(columns=['longitude', 'latitude', 'skipped_updates', 'free_spots'])
    # Deduplisering: ta siste observasjon per time per stasjon
    s = s.sort_values('timestamp').groupby(['station', 'timestamp'], as_index=False).last()
    
    full_hours = pd.date_range(Start_date.floor('h'), end_date.floor('h'), freq='h', tz='UTC')
    s = s.set_index(['station', 'timestamp']).sort_index()
    
    def reindex_ffill_limited(df):
        # LOCF med max 4 timer - dropp gaps over dette
        return df.droplevel(0).reindex(full_hours).ffill(limit=4)
    
    s = s.groupby(level=0).apply(reindex_ffill_limited)
    s.index = s.index.set_names(['station', 'timestamp'])
    s = s.reset_index().dropna(subset=['free_bikes'])  # Fjern rader hvor gaps var >8t
    s['free_bikes'] = s['free_bikes'].astype('int32')
    return s, end_date


def detect_trip_time_columns(cols):
    # Auto-detect kolonnenavn for start/slutt-tidspunkt (robusthet mot ulike formater)
    lc = [c.lower() for c in cols]
    mapping = dict(zip(lc, cols))
    start_candidates = ['started_at', 'start_time', 'starttime', 'start', 'start_timestamp', 'started', 'start_ts']
    end_candidates = ['ended_at', 'end_time', 'endtime', 'end', 'stop_time', 'stop_timestamp', 'stopped', 'end_ts']
    return (next((mapping[c] for c in start_candidates if c in mapping), None),
            next((mapping[c] for c in end_candidates if c in mapping), None))


def compute_hourly_flow(trips_fp, end_date):
    sample = pd.read_csv(trips_fp, nrows=5)
    start_col, end_col = detect_trip_time_columns(sample.columns.tolist())
    if not start_col and not end_col:
        raise ValueError('Could not detect start/end timestamp columns in trips.csv')
    
    parse_ts = lambda s: pd.to_datetime(s, utc=True, errors='coerce').dt.floor('h')
    flow_counts = {}
    
    # Chunked reading (250k rader av gangen) for å håndtere store filer
    for chunk in pd.read_csv(trips_fp, chunksize=250_000):
        # Tell turer som starter i denne timen
        if start_col and start_col in chunk.columns:
            ts = parse_ts(chunk[start_col])
            for h, cnt in ts[(ts >= Start_date) & (ts <= end_date)].value_counts().items():
                flow_counts[h] = flow_counts.get(h, 0) + int(cnt)
        # Tell turer som slutter i denne timen
        if end_col and end_col in chunk.columns:
            te = parse_ts(chunk[end_col])
            for h, cnt in te[(te >= Start_date) & (te <= end_date)].value_counts().items():
                flow_counts[h] = flow_counts.get(h, 0) + int(cnt)
    
    # Lag fullstendig time-grid, timer uten turer blir satt til 0
    full_hours = pd.date_range(Start_date.floor('h'), end_date.floor('h'), freq='h', tz='UTC')
    flow = pd.Series({pd.Timestamp(k): v for k, v in flow_counts.items()}, dtype='float64')
    flow.index = pd.to_datetime(flow.index, utc=True)
    return flow.reindex(full_hours).fillna(0).astype(int).to_frame('flow').reset_index().rename(columns={'index': 'timestamp'})


def build_features(stations_df, weather_df, flow_df):
    # Merge alle datasett på timestamp (inner join sikrer overlapp)
    df = stations_df.merge(weather_df, on='timestamp', how='inner').merge(flow_df, on='timestamp', how='left')
    df['flow'] = df['flow'].fillna(0).astype(int)
    df = df[df['station'].isin(Target_stations)].copy()
    df['timestamp'] = pd.to_datetime(df['timestamp'], utc=True)
    
    # Temporal features
    df['hour'] = df['timestamp'].dt.hour  # 0-23 (rush hour pattern)
    df['is_weekend'] = df['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)  # Lør/søn
    
    # One-hot encode stasjoner (9 kolonner)
    station_dummies = pd.get_dummies(df['station'], prefix='station', dtype=int)
    for name in Target_stations:
        if (col := f'station_{name}') not in station_dummies.columns:
            station_dummies[col] = 0  # Sikkerhet hvis stasjon mangler
    station_dummies = station_dummies[[f'station_{n}' for n in Target_stations]]
    df = pd.concat([df, station_dummies], axis=1)
    
    # Target: free_bikes 1 time fram (shift -1 per stasjon for å unngå lekkasje)
    df = df.sort_values(['station', 'timestamp'])
    df['free_bikes_plus_1h'] = df.groupby('station')['free_bikes'].shift(-1)
    df = df.dropna(subset=['free_bikes_plus_1h']).copy()  # Dropp siste rad per stasjon
    df['free_bikes_plus_1h'] = df['free_bikes_plus_1h'].astype(int)
    
    # Kolonnerekkefølge: station dummies først, deretter features, så target
    final_cols = [f'station_{n}' for n in Target_stations] + [
        'timestamp', 'station', 'free_bikes', 'temperature', 'precipitation',
        'wind_speed', 'flow', 'hour', 'is_weekend', 'free_bikes_plus_1h'
    ]
    df = df[[c for c in final_cols if c in df.columns]]
    return df.sort_values('timestamp').reset_index(drop=True).drop_duplicates(['timestamp', 'station']).dropna()

def my_pipeline(raw_data):
    stations_fp, trips_fp, weather_fp, out_dir = workspace_paths(raw_data)
    # Load stations først for å få dynamisk end_date (viktigste datasett)
    stations_df, end_date = load_stations(stations_fp)
    weather_df = load_weather(weather_fp, end_date)
    flow_df = compute_hourly_flow(trips_fp, end_date)
    model_df = build_features(stations_df, weather_df, flow_df)
    model_df.to_csv(out_dir / 'model_ready.csv', index=False)
    return model_df

if __name__ == '__main__':
    pd.options.mode.chained_assignment = None
    my_pipeline('raw_data')
