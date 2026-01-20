import pandas as pd
import os

data_folder = "data"
os.makedirs(data_folder, exist_ok=True)

files = {
    "Premier League": os.path.join(data_folder, "E0.csv"),
    "Bundesliga": os.path.join(data_folder, "D1.csv")
}

ligas = []

for liga, filepath in files.items():
    print(f"Cargando datos de {liga} desde archivo local...")
    df = pd.read_csv(filepath)

    df = df.rename(columns={
        "HomeTeam": "home",
        "AwayTeam": "away",
        "FTHG": "home_goals",
        "FTAG": "away_goals",
        "FTR": "result"
    })

    ligas.append(df[["home", "away", "home_goals", "away_goals", "result"]])

df_total = pd.concat(ligas, ignore_index=True)
df_total.to_csv("partidos_2023_2024.csv", index=False)

print("\n✅ Archivo creado: partidos_2023_2024.csv")
print(df_total.head())
