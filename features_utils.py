import pandas as pd
import numpy as np
from datetime import datetime


# --- Función auxiliar: last_n_stats (Ahora maneja cualquier 'n') ---
def last_n_stats(df, team, date, n):
    """
    Calcula estadísticas de forma (Goles a favor/en contra, Victorias) de los últimos 'n' partidos.
    Devuelve un diccionario con claves terminadas en '_n'.
    """
    # Filtra partidos pasados para el equipo dado
    past = df[(df["date"] < date) & ((df["home_team"] == team) | (df["away_team"] == team))]
    past = past.tail(n)

    suffix = f"_{n}"

    if past.empty:
        # Devolvemos NaN si no hay historial
        return {f"avg_goals_scored{suffix}": np.nan, f"avg_goals_conceded{suffix}": np.nan, f"wins{suffix}": np.nan}

    goals_for = []
    goals_against = []
    wins = 0

    for _, r in past.iterrows():
        # Extracción de goles
        if r["home_team"] == team:
            gf, ga = r["home_goals"], r["away_goals"]
        else:
            gf, ga = r["away_goals"], r["home_goals"]

        goals_for.append(gf)
        goals_against.append(ga)
        if gf > ga: wins += 1

    return {
        f"avg_goals_scored{suffix}": np.mean(goals_for) if goals_for else np.nan,
        f"avg_goals_conceded{suffix}": np.mean(goals_against) if goals_against else np.nan,
        f"wins{suffix}": wins
    }


# --- Función Principal: compute_features (CORREGIDA) ---
def compute_features(matches_df):
    """
    Calcula features para todos los partidos históricos (usado para el entrenamiento),
    incluyendo la forma para n=3, 5 y 10 partidos.
    """
    df = matches_df.sort_values("date").copy()

    if "home" in df.columns:
        df.rename(columns={"home": "home_team", "away": "away_team"}, inplace=True)

    # --- CÁLCULO DE PROBABILIDAD IMPLÍCITA (Cuotas) ---
    df['B365H'] = pd.to_numeric(df.get('B365H'), errors='coerce')
    df['B365D'] = pd.to_numeric(df.get('B365D'), errors='coerce')
    df['B365A'] = pd.to_numeric(df.get('B365A'), errors='coerce')

    df['prob_implied_H'] = df['B365H'].apply(lambda x: 1 / x if x > 1 else np.nan)
    df['prob_implied_D'] = df['B365D'].apply(lambda x: 1 / x if x > 1 else np.nan)
    df['prob_implied_A'] = df['B365A'].apply(lambda x: 1 / x if x > 1 else np.nan)
    # --------------------------------------------------------

    df["total_goals"] = df["home_goals"] + df["away_goals"]
    df["result"] = df.apply(
        lambda r: "H" if r.home_goals > r.away_goals else ("A" if r.away_goals > r.home_goals else "D"), axis=1)

    # --- CÁLCULO DE FUERZA (PPG) ---
    df['home_points'] = df['result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df['away_points'] = df['result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

    all_home_games = df.groupby('home_team').agg(
        home_points_total=('home_points', 'sum'),
        home_games_total=('home_team', 'count')
    )
    all_away_games = df.groupby('away_team').agg(
        away_points_total=('away_points', 'sum'),
        away_games_total=('away_team', 'count')
    )

    ppg_df = pd.concat([all_home_games, all_away_games], axis=1).fillna(0)
    ppg_df['total_points'] = ppg_df['home_points_total'] + ppg_df['away_points_total']
    ppg_df['total_games'] = ppg_df['home_games_total'] + ppg_df['away_games_total']
    ppg_df['ppg'] = ppg_df['total_points'] / ppg_df['total_games']

    ppg_dict = ppg_df['ppg'].to_dict()
    # ----------------------------------------------------

    # Lista de períodos a calcular
    N_PERIODS = [3, 5, 10]

    rows = []
    for index, r in df.iterrows():
        date = r["date"]
        home = r["home_team"];
        away = r["away_team"]

        row = {
            "index": index,
            "date": date,
            "home_team": home, "away_team": away,
            # Features de Mercado y PPG
            "prob_implied_H": r.get("prob_implied_H"),
            "prob_implied_D": r.get("prob_implied_D"),
            "prob_implied_A": r.get("prob_implied_A"),
            "h_opponent_ppg": ppg_dict.get(away, 1.0),
            "a_opponent_ppg": ppg_dict.get(home, 1.0),
            # Targets
            "result": r["result"],
            "total_goals": r["total_goals"],
        }

        # Calcular Features de Forma para N=3, 5, 10
        for n in N_PERIODS:
            h_stats = last_n_stats(df, home, date, n=n)
            a_stats = last_n_stats(df, away, date, n=n)

            suffix = f"_{n}"

            # Features de Forma LOCAL
            row[f"h_avg_scored{suffix}"] = h_stats[f"avg_goals_scored{suffix}"]
            row[f"h_avg_conceded{suffix}"] = h_stats[f"avg_goals_conceded{suffix}"]
            row[f"h_wins{suffix}"] = h_stats[f"wins{suffix}"]

            # Features de Forma VISITANTE
            row[f"a_avg_scored{suffix}"] = a_stats[f"avg_goals_scored{suffix}"]
            row[f"a_avg_conceded{suffix}"] = a_stats[f"avg_goals_conceded{suffix}"]
            row[f"a_wins{suffix}"] = a_stats[f"wins{suffix}"]

        rows.append(row)

    feat_df = pd.DataFrame(rows).set_index("index")

    # Targets (se mantienen como strings para la limpieza posterior)
    feat_df["target_winner"] = feat_df["result"]
    feat_df["target_under25"] = (feat_df["total_goals"] < 2.5).astype(int)

    return feat_df