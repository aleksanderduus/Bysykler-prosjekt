# Prosjektbeskrivelse INF161 H25: Prognostisering av Bergen Bysykkel

## Oversikt
Målet med prosjektet er å anvende det du har lært i INF161 til å gjennomføre et helhetlig data science-prosjekt. Prosjektet er **obligatorisk** og teller mot sluttkarakteren i emnet.  

Prosjektet er delt i tre hoveddeler, hver med egen frist:  
1. Datautforskning og dataklargjøring – **Frist:** 19.09. 
2. Modellering og prediksjon – **Frist:** 10.10.  
3. Sammenstilling og deployment – **Frist:** 31.10. 

I tillegg gjennomføres to **obligatoriske** fagfellevurderinger: én etter Del 1 og én etter Del 2. Du vil få tildelt en annen students prosjekt å vurdere, og skal gi konstruktive tilbakemeldinger.  

**Viktig:** Alle frister er obligatoriske, men du blir kun karaktervurdert på den endelige innleveringen (**Del 3**). Hensikten er at du gradvis forbedrer prosjektet ditt basert på tilbakemeldinger og økt kunnskap.  

Du burde lese gjennom hele prosjektbeskrivelsen **før** du begynner på Del 1 slik at du gjør riktige antagelser.

## Problemstilling
Du er en data scientist engasjert av en klient med stor interesse for bysykler i Bergen. Klienten ønsker et system som kan:  
- For hver hele klokketime (11:00, 12:00, 13:00 osv.) predikere hvor mange sykler (`free_bikes` kolonnen i `stations.csv`) som vil være tilgjengelige på en stasjon én time senere, basert på data tilgjengelig på prediksjonstidspunktet.  
- Vi bryr oss bare om å predikere følgende stasjoner:  
  - `Møllendalsplass`  
  - `Torgallmenningen`  
  - `Grieghallen`  
  - `Høyteknologisenteret`  
  - `Studentboligene`  
  - `Akvariet`  
  - `Damsgårdsveien 71`  
  - `Dreggsallmenningen Sør`  
  - `Florida Bybanestopp`  

Men du kan bruke historiske og samtidige data fra alle stasjoner, bare ikke fremtidsdata.

Det endelige målet er å utvikle en pipeline som kan kjøres på nye data og gi prediksjoner til klienten.  

### Tilgjengelige data
Dataene finnes i `raw_data`-mappen og består av tre dataframes. En detaljert beskrivelse ligger i `raw_data/README.md`. Klienten mottar kontinuerlig nye observasjoner, og systemet skal kunne kjøres løpende for å produsere oppdaterte prediksjoner.

### Viktige antakelser
For å lage labels antar vi at sykler på en stasjon tilsvarer den siste registrerte verdien fram til en ny observasjon foreligger (*Last Observation Carried Forward/LOCF*). Eksempel: Dersom siste måling kl. `16:49:25` viser 8 sykler, antar vi at tallet gjelder frem til neste oppdatering kl. `18:00:02` som viser 10 sykler. Når du da predikerer på tidspunkt `17:00:00` hva sykkeltilgjengeligheten er på tidspunkt `18:00:00` er det riktige svaret 8 sykler.


---

# Del 1: Datautforskning og dataklargjøring (40 poeng)

**Leveranser:**  
- Kode for datautforskning.  
- Kode for dataklargjøring.  
- Rapport som beskriver metode og funn.  

**Oppgaver:**  
1. Utforsk og analyser dataene med visualiseringer og beskrivende statistikk.  
2. Utvikle en pipeline som omgjør rådata til en samlet dataframe, `model_ready.csv`, med én prediksjonslabel per rad. Hver rad skal inneholde både label og relevante features.  
3. Lag en kort rapport som dokumenterer:  
   - Fremgangsmåte og begrunnelser  
   - Dataegenskaper (struktur, feil, mangler, innsikter)
   - Relevante visualiseringer  
   - Preprosesseringstiltak og valg 

**Pipeline:**  
- **Input:**  
  - `stations.csv`  
  - `trips.csv`  
  - `weather.csv`  
- **Output:**  
  - `model_ready.csv` – klargjort for Del 2  

**Tips:**  
- Bruk Jupyter Notebooks til utforskning og prototyping, men migrer ferdig pipeline til en `.py`-fil.  
- Sørg for at resultatene kan reproduseres, dette kan involvere å sette random seeds.  
- Pass på å unngå tidslekkasjer.  
- Du må muligens være litt kreativ i hvordan du utnytter `trips.csv`.  

  ```  

---

# Del 2: Modellering og prediksjon (40 poeng)

**Leveranser:**  
- Kode.  
- Rapport.  
- En lagret ML-modell.  

**Oppgaver:**  
- Tren en maskinlæringsmodell basert på `model_ready.csv`.  
- Prøv minst tre ulike modelltyper før du velger endelig modell.  
- Dokumenter hva du har gjort med begrunnelser. Feks: håndtering av features, manglende data og eventuelle imputeringsteknikker, modelvalg, oppsplitting av data, etc. 
- Evaluer ytelse ved å bruke **root mean square error (RMSE):**  


$\text{RMSE} = \sqrt{\frac{\sum_{i=1}^N (\hat{y}_i - y_i)^2}{N}}$
 
Hvor $N$ er antall prediksjoner, $\hat{y}_i$ er predikert verdi, og $y_i$ er faktisk verdi.

- Rapporter forventet generaliseringsytelse (RMSE) og begrunn modellvalget.  

**Pipeline:**  
- **Input:** `model_ready.csv`  
- **Output:**  
  1. Lagret modell  
  2. Forventet generaliseringsfeil (RMSE)  

**Tips:**
- Det går helt fint om du oppdaterer pipelinen din fra **Del 1.** og genererer ny `model_ready.csv`.
- Husk at vi bare bryr oss om å predikere delmengden av stasjoner som er ramset opp i problemstillingen, og modelvalg/generaliseringsytelse skal reflektere dette.
- Husk standard data-science metodologi for automatisk modellutvelgelse og evaluering.

---

# Del 3: Sammenstilling og deployment (20 poeng)

Når pipelinen fungerer, skal den gjøres robust og tilgjengelig for klienten via to skript:  

1. **`train.py`** – kjører hele pipelinen fra rådata til lagret modell, inkludert modellvalg og evaluering.  
   - **Input:** rådata  
   - **Output:** lagret modell + forventet RMSE  

2. **`predict.py`** – bruker den lagrede modellen til å lage prediksjoner for neste hele klokketime (her antar vi at det nyeste tidsstempelet som finnes i noen av de 3 *rå* datarammene rundet opp til neste hele klokketime er tidspunktet vi predikerer fra).  
   - **Input:** rådata + lagret modell  
   - **Output:**  
     - Siste tidsstempel i dataene (gjort om fra UTC til lokal Bergen-tid slik at klienten ikke blir forvirret)  
     - Tidsstempel for neste hele klokketime.
     - Tidsstempelet prediksjonen gjelder for  
     - Nåværende og predikerte antall sykler og prediksjoner for alle målstasjoner  
     
Skriptene skal være satt opp slik at man kan kjøre `python <script_name>.py` gitt at man er i rotmappen i prosjektet ditt. Koden skal fungere selv om det legges til nye rader i de råe datarammene. 

**Eksempel på utskrift:**  

- Siste tidsstempel i data: `2025-04-07 13:47:01+02:00` 
- Neste hele klokketime: `2025-04-07 14:00:00+02:00`
- Predikerer for tidsstempel: `2025-04-07 15:00:00+02:00`  

| Stasjon                | Nåværende sykler | Predikerte sykler |  
|-------------------------|------------------|-------------------|  
| Møllendalsplass         | 12               | 10                |  
| Torgallmenningen        | 8                | 9                 |  
| Grieghallen             | 5                | 7                 |  
| Høyteknologisenteret    | 6                | 8                 |  
| Studentboligene         | 4                | 5                 |  
| Akvariet                | 7                | 6                 |  
| Damsgårdsveien 71       | 3                | 4                 |  
| Dreggsallmenningen Sør  | 9                | 10                |  
| Florida Bybanestopp     | 11               | 12                |  


---

## Endelige leveranser
- **PDF-rapport** med beskrivelse av metode, resultater, analyser og refleksjoner. Kvaliteten på rapporten er del av vurdering. 
- **Zip-fil** med kode, modellfil og nødvendige skript (`train.py`, `predict.py`) samt en `hvordan_bruke.txt` som forklarer hvordan man kjører koden din om det ikke er åpenbart.  

**Krav:**  
- Koden skal være godt dokumentert.  
- Resonnement skal fremgå tydelig i rapporten.  
- **Tillate biblioteker**: Python standard library + `folium`, `numpy`, `pandas`, `polars`, `scipy`, `sklearn`, `matplotlib`, `seaborn`, `plotly`, `xgboost`. 

## Bruk av KI

Det er lov å bruke KI men bruken må rapporteres. Følg retningslinjene fra fakultetet: https://www.uib.no/nt/177637/bruk-av-kunstig-intelligens-ki-ved-fakultet-naturvitenskap-og-teknologi-universitetet-i

## Datakilder

Værdata fra Open meteo (https://open-meteo.com/) publisert under CC BY 4.0

Bergen bysykkel data (https://bergenbysykkel.no/en/open-data) publisert under Norsk lisens for offentlige data (NLOD) 2.0 (https://data.norge.no/nlod/no/2.0)

https://github.com/MaxHalford/bike-sharing-history