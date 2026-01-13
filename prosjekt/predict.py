import pandas as pd
from datetime import timedelta
from My_Pipeline import my_pipeline
import joblib
import numpy as np
import pandas as pd

df = my_pipeline('raw_data')

#Hent siste timestamp og beregn neste klokkeslett
last_ts = pd.to_datetime(df['timestamp']).max().tz_convert('Europe/Oslo')
pred_hour = last_ts + timedelta(hours=1)

model = joblib.load("best_model.pkl")

#Velg siste tilgjengelige rader og prediker
latest = df.sort_values("timestamp").groupby("station").tail(1).copy()
X_latest = latest.drop(columns=['timestamp', 'station', 'free_bikes_plus_1h'])
latest["predicted_free_bikes"] = model.predict(X_latest)

#Skriv ut resultatet i ønsket format
print(f"Siste tidsstempel i data: {last_ts}")
print(f"Predikerer for tidsstempel: {pred_hour}\n")

print(f"{'Stasjon':<25} {'Nåværende sykler':<18} {'Predikerte sykler':<20}")
print("-" * 60)
for _, row in latest.iterrows():
    print(f"{row['station']:<25} {row['free_bikes']:<18} {row['predicted_free_bikes']:<20.0f}")
