import os
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import log_loss
from sklearn.model_selection import GridSearchCV
from xgboost import XGBClassifier
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

from features_utils import compute_features, last_n_stats

# --- CONSTANTES DE FEATURE SETS (SIMPLIFICADAS A n=10) ---
N10_FEATURES = [
    "h_avg_scored_10", "h_avg_conceded_10", "h_wins_10",
    "a_avg_scored_10", "a_avg_conceded_10", "a_wins_10",
]
FULL_FEATURES_LIST = N10_FEATURES
WINNER_FEATURES_LIST = N10_FEATURES


# --- FUNCIÓN: Optimización de Hiperparámetros ---
def optimize_xgb_hyperparameters(X, y, model_name="Modelo"):
    print(f"\nIniciando optimización de hiperparámetros para {model_name}...")

    param_grid = {
        'n_estimators': [100, 300, 500],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7]
    }

    objective = 'multi:softmax' if model_name == "Ganador (H/D/A)" else 'binary:logistic'
    num_class = 3 if model_name == "Ganador (H/D/A)" else None
    eval_metric = 'mlogloss' if model_name == "Ganador (H/D/A)" else 'logloss'

    xgb = XGBClassifier(
        objective=objective, num_class=num_class, use_label_encoder=False,
        eval_metric=eval_metric, random_state=42, tree_method='hist'
    )

    grid_search = GridSearchCV(
        estimator=xgb, param_grid=param_grid, scoring='neg_log_loss',
        cv=5, verbose=0, n_jobs=-1
    )
    grid_search.fit(X, y)

    print(f"Mejores hiperparámetros: {grid_search.best_params_}")
    print(f"Mejor Log Loss: {-grid_search.best_score_:.4f}")

    clf_optimized = XGBClassifier(
        objective=objective, num_class=num_class, use_label_encoder=False,
        eval_metric=eval_metric, random_state=42, tree_method='hist',
        **grid_search.best_params_
    )
    clf_optimized.fit(X, y)
    return clf_optimized


# --- FUNCIÓN: Visualizar Feature Importance ---
def plot_feature_importance(model, feature_names):
    importance = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names, 'Importance': importance
    }).sort_values(by='Importance', ascending=True)

    plt.figure(figsize=(10, 6))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
    plt.xlabel('Importancia (F Score)', fontsize=12)
    plt.title('Importancia de las Características (Goles < 2.5) - Modelo n=10', fontsize=14)
    plt.tight_layout()
    plt.savefig("data/feature_importance_xgboost_n10.png")
    plt.close()


# --- Model training ---
def train_models(features_df, y_winner_encoded, y_under):
    X_winner = features_df[WINNER_FEATURES_LIST]
    print(f"\nEntrenando Modelo de Ganador con {len(X_winner.columns)} features...")
    clf_winner = optimize_xgb_hyperparameters(X_winner, y_winner_encoded, model_name="Ganador (H/D/A)")

    X_under = features_df[FULL_FEATURES_LIST]
    print(f"\nEntrenando Modelo de Goles con {len(X_under.columns)} features...")
    clf_under = optimize_xgb_hyperparameters(X_under, y_under, model_name="Goles (Under/Over)")

    plot_feature_importance(clf_under, FULL_FEATURES_LIST)
    return clf_winner, clf_under


# --- Generación de Predicciones Históricas ---
def generate_historical_predictions(features_df, df_original, model_winner, model_under, y_winner_encoded, y_under):
    X_winner = features_df[WINNER_FEATURES_LIST]
    probs_winner = model_winner.predict_proba(X_winner)

    X_under = features_df[FULL_FEATURES_LIST]
    probs_under = model_under.predict_proba(X_under)[:, 1]

    df_preds = df_original.copy().reset_index(drop=True)
    df_preds["target_winner_encoded"] = y_winner_encoded.reset_index(drop=True)
    df_preds["target_under25"] = y_under.reset_index(drop=True)
    df_preds["prob_H"] = probs_winner[:, 0]
    df_preds["prob_D"] = probs_winner[:, 1]
    df_preds["prob_A"] = probs_winner[:, 2]
    df_preds["prob_Under25"] = probs_under
    df_preds["prob_Over25"] = 1 - probs_under
    df_preds.rename(columns={"result": "target_result"}, inplace=True)

    df_features_limpias = features_df.reset_index(drop=True)
    cols_to_concat = WINNER_FEATURES_LIST
    df_preds = pd.concat([df_preds, df_features_limpias[cols_to_concat]], axis=1)
    return df_preds.dropna(subset=["home_team", "away_team"])


# --- Función auxiliar para calcular PPG ---
def calculate_ppg_dict(df):
    df['home_points'] = df['result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df['away_points'] = df['result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

    all_home = df.groupby('home_team').agg(home_points_total=('home_points', 'sum'),
                                           home_games_total=('home_team', 'count'))
    all_away = df.groupby('away_team').agg(away_points_total=('away_points', 'sum'),
                                           away_games_total=('away_team', 'count'))

    ppg_df = pd.concat([all_home, all_away], axis=1).fillna(0)
    ppg_df['total_points'] = ppg_df['home_points_total'] + ppg_df['away_points_total']
    ppg_df['total_games'] = ppg_df['home_games_total'] + ppg_df['away_games_total']
    ppg_df['ppg'] = ppg_df['total_points'] / ppg_df['total_games']
    return ppg_df['ppg'].to_dict()


# --- Función para predecir partido futuro ---
def get_prediction_features(df_full, ppg_dict, home_team, away_team, odds_H=1.0, odds_D=1.0, odds_A=1.0):
    df_temp = df_full.copy()
    date_now = datetime.now()
    ALL_STATS = {}
    N_PERIODS = [3, 5, 10]

    for n in N_PERIODS:
        h_stats = last_n_stats(df_temp, home_team, date_now, n=n)
        a_stats = last_n_stats(df_temp, away_team, date_now, n=n)
        suffix = f"_{n}"
        ALL_STATS.update({
            f"h_avg_scored{suffix}": h_stats.get(f"avg_goals_scored{suffix}"),
            f"h_avg_conceded{suffix}": h_stats.get(f"avg_goals_conceded{suffix}"),
            f"h_wins{suffix}": h_stats.get(f"wins{suffix}"),
            f"a_avg_scored{suffix}": a_stats.get(f"avg_goals_scored{suffix}"),
            f"a_avg_conceded{suffix}": a_stats.get(f"avg_goals_conceded{suffix}"),
            f"a_wins{suffix}": a_stats.get(f"wins{suffix}")
        })

    prob_H = 1 / odds_H if odds_H >= 1 else np.nan
    prob_D = 1 / odds_D if odds_D >= 1 else np.nan
    prob_A = 1 / odds_A if odds_A >= 1 else np.nan
    h_opponent_ppg = ppg_dict.get(away_team, 1.0)
    a_opponent_ppg = ppg_dict.get(home_team, 1.0)

    features_dict = {**ALL_STATS, "prob_implied_H": prob_H, "prob_implied_D": prob_D,
                     "prob_implied_A": prob_A, "h_opponent_ppg": h_opponent_ppg,
                     "a_opponent_ppg": a_opponent_ppg}

    X_winner = np.array([features_dict.get(f, np.nan) for f in WINNER_FEATURES_LIST]).reshape(1, -1)
    X_under = np.array([features_dict.get(f, np.nan) for f in FULL_FEATURES_LIST]).reshape(1, -1)

    if X_winner.shape[1] != 6 or X_under.shape[1] != 6:
        return None, None

    if np.any(np.isnan(X_winner)) or np.any(np.isnan(X_under)):
        return None, None

    return X_winner, X_under


# --- EJECUCIÓN PRINCIPAL ---
if __name__ == "__main__":
    # Carga de datos
    archivos_historicos = [
        "data/partidos_2023_2024.csv",
        "data/D1.csv", "data/E0.csv", "data/SP1.csv", "data/I1.csv"
    ]
    lista_df = []

    for ruta_archivo in archivos_historicos:
        try:
            df = pd.read_csv(ruta_archivo, encoding='latin-1', on_bad_lines='skip')
            df.rename(columns={
                "HomeTeam": "home_team", "AwayTeam": "away_team",
                "FTHG": "home_goals", "FTAG": "away_goals",
                "Date": "date", "home": "home_team", "away": "away_team"
            }, inplace=True, errors='ignore')

            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"], errors='coerce', dayfirst=True)
                df.dropna(subset=["date", "home_team"], inplace=True)
            lista_df.append(df)
        except FileNotFoundError:
            print(f"Advertencia: '{ruta_archivo}' no encontrado. Saltando.")
        except Exception as e:
            print(f"Error al procesar '{ruta_archivo}': {e}")

    if not lista_df:
        print("Error fatal: No se pudo cargar ningún archivo histórico.")
        exit()

    matches_df = pd.concat(lista_df, ignore_index=True)
    columnas_requeridas = ["date", "home_team", "away_team", "home_goals", "away_goals"]
    matches_df = matches_df.dropna(subset=columnas_requeridas).sort_values("date")
    matches_df["result"] = matches_df.apply(
        lambda r: "H" if r.home_goals > r.away_goals else ("A" if r.away_goals > r.home_goals else "D"), axis=1)

    print(f"Total de partidos combinados: {len(matches_df)}")
    print("Calculando features...")
    features = compute_features(matches_df)

    # Preparar targets y limpieza
    y_winner_encoded = features["target_winner"].map({'H': 0, 'D': 1, 'A': 2}).rename('y_winner_encoded')
    y_under = features["target_under25"].rename('y_under')

    all_features = list(set(FULL_FEATURES_LIST))
    columns_to_check = all_features + ['y_winner_encoded', 'y_under']

    df_temp_clean = pd.concat([features[all_features], y_winner_encoded, y_under], axis=1)
    df_clean = df_temp_clean.dropna(subset=columns_to_check).copy()

    features_clean_final = df_clean[all_features]
    y_winner_clean_final = df_clean['y_winner_encoded'].astype(int)
    y_under_clean_final = df_clean['y_under']

    print(f"Filas originales: {len(features)}. Filas limpias: {len(features_clean_final)}")

    if len(features_clean_final) == 0:
        print("Error: No hay datos limpios suficientes para entrenar.")
        exit()

    print(f"Entrenando modelos con {len(features_clean_final)} filas...")
    model_winner, model_under = train_models(features_clean_final, y_winner_clean_final, y_under_clean_final)

    # Guardar modelos en RAÍZ DEL PROYECTO
    joblib.dump(model_winner, "model_winner_n10.joblib")
    joblib.dump(model_under, "model_under_n10.joblib")

    # Generar y guardar predicciones históricas
    df_original_clean = matches_df.loc[features_clean_final.index]
    df_preds_hist = generate_historical_predictions(
        features_clean_final, df_original_clean, model_winner, model_under,
        y_winner_clean_final, y_under_clean_final
    )

    df_preds_hist.to_csv("data/historical_preds_n10.csv", index=False)
    print("\n✅ Modelos guardados en RAÍZ y predicciones históricas generadas en data/")