import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime
from sklearn.metrics import accuracy_score, log_loss
from features_utils import last_n_stats
import os

# --- CONSTANTES ---
N10_FEATURES = [
    "h_avg_scored_10", "h_avg_conceded_10", "h_wins_10",
    "a_avg_scored_10", "a_avg_conceded_10", "a_wins_10",
]
FULL_FEATURES_LIST = N10_FEATURES
WINNER_FEATURES_LIST = N10_FEATURES

# Mapeo de archivos a ligas
LEAGUE_MAPPING = {
    "E0.csv": "Premier League", "D1.csv": "Bundesliga",
    "SP1.csv": "La Liga", "I1.csv": "Serie A",
    "partidos_2023_2024.csv": "Mixto"
}


@st.cache_data
def load_data():
    try:
        df_hist = pd.read_csv("data/historical_preds_n10.csv")
        model_winner = joblib.load("model_winner_n10.joblib")
        model_under = joblib.load("model_under_n10.joblib")

        if 'target_result' in df_hist.columns:
            df_hist.rename(columns={'target_result': 'target_winner'}, inplace=True)
        elif 'result' in df_hist.columns and 'target_winner' not in df_hist.columns:
            df_hist.rename(columns={'result': 'target_winner'}, inplace=True)

        df_hist["date"] = pd.to_datetime(df_hist["date"])

        archivos_historicos = [
            "data/partidos_2023_2024.csv", "data/D1.csv", "data/E0.csv",
            "data/SP1.csv", "data/I1.csv"
        ]
        lista_df_raw = []

        for ruta_archivo in archivos_historicos:
            try:
                df = pd.read_csv(ruta_archivo, encoding='latin-1', on_bad_lines='skip')
                df = df.rename(columns={
                    "HomeTeam": "home_team", "AwayTeam": "away_team",
                    "FTHG": "home_goals", "FTAG": "away_goals", "Date": "date",
                    "home": "home_team", "away": "away_team"
                }, errors='ignore')

                if "date" in df.columns:
                    df["date"] = pd.to_datetime(df["date"], errors='coerce', dayfirst=True)
                    df.dropna(subset=["date", "home_team"], inplace=True)

                nombre_archivo = os.path.basename(ruta_archivo)
                df["league"] = LEAGUE_MAPPING.get(nombre_archivo, "Desconocida")
                lista_df_raw.append(df)
            except:
                pass

        df_raw_full = pd.concat(lista_df_raw, ignore_index=True)
        df_raw_full["date"] = pd.to_datetime(df_raw_full["date"], errors='coerce')
        df_raw_full["result"] = df_raw_full.apply(
            lambda r: "H" if r.home_goals > r.away_goals else ("A" if r.away_goals > r.home_goals else "D"), axis=1
        )

        return df_hist, model_winner, model_under, df_raw_full

    except Exception as e:
        st.error(f"Error al cargar datos: {e}")
        return None, None, None, None


@st.cache_data
def calculate_ppg_dict(df):
    df['home_points'] = df['result'].apply(lambda x: 3 if x == 'H' else (1 if x == 'D' else 0))
    df['away_points'] = df['result'].apply(lambda x: 3 if x == 'A' else (1 if x == 'D' else 0))

    all_home = df.groupby('home_team').agg(
        home_points_total=('home_points', 'sum'),
        home_games_total=('home_team', 'count')
    )
    all_away = df.groupby('away_team').agg(
        away_points_total=('away_points', 'sum'),
        away_games_total=('away_team', 'count')
    )

    ppg_df = pd.concat([all_home, all_away], axis=1).fillna(0)
    ppg_df['total_points'] = ppg_df['home_points_total'] + ppg_df['away_points_total']
    ppg_df['total_games'] = ppg_df['home_games_total'] + ppg_df['away_games_total']
    ppg_df['ppg'] = ppg_df['total_points'] / ppg_df['total_games']
    return ppg_df['ppg'].to_dict()


def save_manual_prediction(home, away, league, probs_winner, prob_under):
    filepath = "data/manual_predictions.csv"
    df_new = pd.DataFrame([{
        'match_id': f"{home}_{away}_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
        'date': datetime.now(),
        'home_team': home,
        'away_team': away,
        'league': league,
        'prob_H': probs_winner[0],
        'prob_D': probs_winner[1],
        'prob_A': probs_winner[2],
        'prob_Under25': prob_under,
        'actual_result': None,
        'actual_goals': None,
        'actual_under25': None,
        'status': 'pending'
    }])

    try:
        df_existing = pd.read_csv(filepath)
        df_combined = pd.concat([df_existing, df_new], ignore_index=True)
    except FileNotFoundError:
        df_combined = df_new

    df_combined.to_csv(filepath, index=False)
    return df_new['match_id'].iloc[0]


def load_manual_predictions():
    try:
        df = pd.read_csv("data/manual_predictions.csv")
        df['date'] = pd.to_datetime(df['date'])
        return df
    except FileNotFoundError:
        return pd.DataFrame(columns=[
            'match_id', 'date', 'home_team', 'away_team', 'league',
            'prob_H', 'prob_D', 'prob_A', 'prob_Under25',
            'actual_result', 'actual_goals', 'actual_under25', 'status'
        ])


def update_manual_result(match_id, actual_result, home_goals, away_goals):
    df = load_manual_predictions()
    idx = df[df['match_id'] == match_id].index

    if len(idx) > 0:
        total_goals = home_goals + away_goals
        df.loc[idx[0], 'actual_result'] = actual_result
        df.loc[idx[0], 'actual_goals'] = f"{home_goals}-{away_goals}"
        df.loc[idx[0], 'actual_under25'] = 1 if total_goals < 2.5 else 0
        df.loc[idx[0], 'status'] = 'completed'
        df.to_csv("data/manual_predictions.csv", index=False)
        return True
    return False


def delete_manual_prediction(match_id):
    df = load_manual_predictions()
    df = df[df['match_id'] != match_id]
    df.to_csv("data/manual_predictions.csv", index=False)


def calculate_manual_backtesting():
    df = load_manual_predictions()
    df_completed = df[df['status'] == 'completed'].copy()

    if len(df_completed) < 5:
        return None, "Necesitas al menos 5 partidos completados"

    y_true_winner = df_completed['actual_result']
    y_pred_winner = df_completed[['prob_H', 'prob_D', 'prob_A']].idxmax(axis=1).str.replace('prob_', '')
    acc_winner = accuracy_score(y_true_winner, y_pred_winner)
    logloss_winner = log_loss(y_true_winner.map({'H': 0, 'D': 1, 'A': 2}),
                              df_completed[['prob_H', 'prob_D', 'prob_A']])

    y_true_under = df_completed['actual_under25']
    y_pred_under = (df_completed['prob_Under25'] > 0.5).astype(int)
    acc_under = accuracy_score(y_true_under, y_pred_under)
    logloss_under = log_loss(y_true_under, df_completed['prob_Under25'])

    return {
        'total_matches': len(df_completed),
        'accuracy_winner': acc_winner,
        'accuracy_under': acc_under,
        'logloss_winner': logloss_winner,
        'logloss_under': logloss_under,
        'df': df_completed
    }, None


def calculate_metrics_by_league():
    df = load_manual_predictions()
    df_completed = df[df['status'] == 'completed'].copy()

    if len(df_completed) < 5:
        return None, "Necesitas al menos 5 partidos completados"

    results = []
    for league in df_completed['league'].unique():
        df_liga = df_completed[df_completed['league'] == league]

        if len(df_liga) < 3:
            continue

        y_true_winner = df_liga['actual_result']
        y_pred_winner = df_liga[['prob_H', 'prob_D', 'prob_A']].idxmax(axis=1).str.replace('prob_', '')
        acc_winner = accuracy_score(y_true_winner, y_pred_winner)

        y_true_under = df_liga['actual_under25']
        y_pred_under = (df_liga['prob_Under25'] > 0.5).astype(int)
        acc_under = accuracy_score(y_true_under, y_pred_under)

        results.append({
            'Liga': league,
            'Partidos': len(df_liga),
            'Precisión H/D/A Numérica': acc_winner,
            'Precisión Under/Over Numérica': acc_under,
            'Precisión H/D/A': f"{acc_winner:.2%}",
            'Precisión Under/Over': f"{acc_under:.2%}"
        })

    return pd.DataFrame(results), None


def get_team_history(df, team, n=9):
    df_filtered = df[(df["home_team"] == team) | (df["away_team"] == team)].dropna(
        subset=['home_goals', 'away_goals', 'date']).sort_values("date").tail(n)

    def get_result(row, team):
        if row["home_team"] == team:
            return "W" if row["home_goals"] > row["away_goals"] else (
                "L" if row["home_goals"] < row["away_goals"] else "D")
        else:
            return "W" if row["away_goals"] > row["home_goals"] else (
                "L" if row["away_goals"] < row["home_goals"] else "D")

    df_filtered['resultado_equipo'] = df_filtered.apply(lambda r: get_result(r, team), axis=1)
    df_filtered['date'] = df_filtered['date'].dt.strftime('%Y-%m-%d')

    return df_filtered[["date", "home_team", "away_team", "home_goals", "away_goals", "resultado_equipo"]].rename(
        columns={'resultado_equipo': 'Resultado'})


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


st.set_page_config(page_title="Football Prediction Bot", layout="wide")
st.title("⚽ Football Prediction Bot - Sistema de Tracking")

df_hist_full, model_winner, model_under, df_raw = load_data()

if df_hist_full is not None and model_winner is not None:
    all_teams = sorted(df_hist_full["home_team"].unique())
    ppg_dict_full = calculate_ppg_dict(df_raw)

    # BARRA LATERAL
    st.sidebar.header("🎯 Tracking de Predicciones")

    with st.sidebar.form("nuevo_partido"):
        st.subheader("Agregar Partido")
        home_manual = st.selectbox("Local", all_teams)
        away_manual = st.selectbox("Visitante", all_teams, index=1)
        league_manual = st.selectbox("Liga", ["Premier League", "Bundesliga", "La Liga", "Serie A", "Mixto"])

        if st.form_submit_button("Agregar a Tracking"):
            X_w, X_u = get_prediction_features(df_raw, ppg_dict_full, home_manual, away_manual)
            if X_w is not None:
                proba_winner = model_winner.predict_proba(X_w)[0]
                proba_under = model_under.predict_proba(X_u)[0][1]
                match_id = save_manual_prediction(home_manual, away_manual, league_manual, proba_winner, proba_under)
                st.sidebar.success(f"✅ ID: {match_id}")
                st.rerun()
            else:
                st.sidebar.error("❌ No hay historial suficiente")

    # HISTORIAL DE EQUIPOS
    st.header("📈 Historial de Equipos")

    col_team1, col_team2 = st.columns(2)

    with col_team1:
        team_history_select = st.selectbox("Selecciona Equipo a Analizar", all_teams, index=0)

    with col_team2:
        n_games = st.slider("Número de partidos a mostrar", min_value=5, max_value=20, value=9, step=1)

    df_history = get_team_history(df_raw, team_history_select, n=n_games)

    if not df_history.empty:
        st.dataframe(df_history, use_container_width=True, height=400)

        col_stats1, col_stats2, col_stats3 = st.columns(3)
        wins = (df_history['Resultado'] == 'W').sum()
        draws = (df_history['Resultado'] == 'D').sum()
        losses = (df_history['Resultado'] == 'L').sum()

        col_stats1.metric("Victorias", wins)
        col_stats2.metric("Empates", draws)
        col_stats3.metric("Derrotas", losses)
    else:
        st.warning(f"No hay datos históricos suficientes para {team_history_select}")

    st.divider()

    # PESTAÑAS
    tab1, tab2, tab3, tab4 = st.tabs(
        ["📊 Dashboard", "🎯 Mis Predicciones", "📈 Análisis por Liga", "⚙️ Predicción Rápida"])

    with tab1:
        st.subheader("Métricas de Rendimiento")
        col1, col2, col3, col4 = st.columns(4)

        y_true_winner = df_hist_full["target_winner"]
        df_hist_full["pred_winner"] = df_hist_full[["prob_H", "prob_D", "prob_A"]].idxmax(axis=1).str.replace('prob_',
                                                                                                              '')
        acc_winner_hist = accuracy_score(y_true_winner, df_hist_full["pred_winner"])

        y_true_under = df_hist_full["target_under25"]
        df_hist_full["pred_under"] = (df_hist_full["prob_Under25"] > 0.5).astype(int)
        acc_under_hist = accuracy_score(y_true_under, df_hist_full["pred_under"])

        col1.metric("Precisión Histórica H/D/A", f"{acc_winner_hist:.2%}")
        col2.metric("Precisión Histórica Under/Over", f"{acc_under_hist:.2%}")

        metrics_manual, error_manual = calculate_manual_backtesting()
        if metrics_manual:
            col3.metric("Mis Predicciones", f"{metrics_manual['total_matches']} partidos")
            col4.metric("Mi Precisión H/D/A", f"{metrics_manual['accuracy_winner']:.2%}")
        else:
            col3.metric("Mis Predicciones", "0 partidos")
            col4.info("Agrega partidos")

    with tab2:
        st.subheader("Gestión de Mis Predicciones")

        df_manual = load_manual_predictions()

        if df_manual.empty:
            st.info("No hay predicciones manuales. Agrega una desde la barra lateral.")
        else:
            status_filter = st.multiselect("Filtrar por Estado",
                                           options=['pending', 'completed'],
                                           default=['pending', 'completed'])
            df_filtered = df_manual[df_manual['status'].isin(status_filter)]

            st.dataframe(df_filtered[['date', 'home_team', 'away_team', 'league',
                                      'prob_H', 'prob_D', 'prob_A', 'prob_Under25',
                                      'actual_result', 'status']].round(3),
                         use_container_width=True)

            st.subheader("✏️ Actualizar Resultado")
            col_upd1, col_upd2, col_upd3, col_upd4, col_upd5 = st.columns([2, 1, 1, 1, 1])

            with col_upd1:
                match_to_update = st.selectbox("Seleccionar Partido",
                                               options=df_manual[df_manual['status'] == 'pending']['match_id'].tolist())

            with col_upd2:
                home_score = st.number_input("Goles Local", min_value=0, max_value=15, step=1)

            with col_upd3:
                away_score = st.number_input("Goles Visitante", min_value=0, max_value=15, step=1)

            with col_upd4:
                actual_result = st.selectbox("Resultado", ["H", "D", "A"])

            with col_upd5:
                st.write("")
                st.write("")
                if st.button("Guardar Resultado", type="primary"):
                    if match_to_update:
                        success = update_manual_result(match_to_update, actual_result, home_score, away_score)
                        if success:
                            st.success("✅ Resultado actualizado")
                            st.rerun()
                        else:
                            st.error("❌ Error al actualizar")

            st.subheader("🗑️ Eliminar Predicción")
            match_to_delete = st.selectbox("Seleccionar Partido a Eliminar",
                                           options=df_manual['match_id'].tolist())
            if st.button("Eliminar Predicción", type="secondary"):
                delete_manual_prediction(match_to_delete)
                st.success("✅ Predicción eliminada")
                st.rerun()

            if st.button("📊 Analizar Mis Predicciones"):
                metrics, error = calculate_manual_backtesting()
                if error:
                    st.warning(error)
                else:
                    st.success(f"Análisis completado: {metrics['total_matches']} partidos")
                    col_a, col_b = st.columns(2)
                    col_a.metric("Precisión H/D/A", f"{metrics['accuracy_winner']:.2%}")
                    col_b.metric("Precisión Under/Over", f"{metrics['accuracy_under']:.2%}")

                    # ==== NUEVO: TABLA CON COLORES (CORREGIDO) ====
                    with st.expander("🔍 Ver Detalles con Aciertos/Fallos"):
                        # Preparar dataframe para mostrar - AHORA SÍ INCLUYE actual_under25
                        df_detalle = metrics['df'][['date', 'home_team', 'away_team', 'league',
                                                    'prob_H', 'prob_D', 'prob_A', 'prob_Under25',
                                                    'actual_result', 'actual_goals', 'actual_under25']].round(3).copy()

                        # Calcular predicciones
                        df_detalle['Pred_HDA'] = df_detalle[['prob_H', 'prob_D', 'prob_A']].idxmax(axis=1).str.replace(
                            'prob_', '')
                        df_detalle['Pred_UO'] = (df_detalle['prob_Under25'] > 0.5).map({True: 'Under', False: 'Over'})

                        # Calcular aciertos
                        acierto_hda = df_detalle['Pred_HDA'] == df_detalle['actual_result']
                        acierto_uo = df_detalle['Pred_UO'] == df_detalle['actual_under25']

                        # Crear columnas de estado con colores
                        df_detalle['Estado H/D/A'] = acierto_hda.map({
                            True: '🟢 ACIERTO',
                            False: '🔴 FALLO'
                        })
                        df_detalle['Estado U/O'] = acierto_uo.map({
                            True: '🟢 ACIERTO',
                            False: '🔴 FALLO'
                        })

                        # Mostrar tabla con colores
                        st.table(df_detalle[['date', 'home_team', 'away_team', 'league',
                                             'prob_H', 'prob_D', 'prob_A', 'prob_Under25',
                                             'actual_result', 'actual_goals',
                                             'Estado H/D/A', 'Estado U/O']])

    with tab3:
        st.subheader("Rendimiento por Liga (Mis Predicciones)")

        metrics_by_league, error = calculate_metrics_by_league()
        if error:
            st.warning(error)
        else:
            display_df = metrics_by_league[['Liga', 'Partidos', 'Precisión H/D/A', 'Precisión Under/Over']].copy()
            st.dataframe(display_df, use_container_width=True)

            if not metrics_by_league.empty:
                import plotly.express as px

                fig = px.bar(metrics_by_league, x='Liga',
                             y=['Precisión H/D/A Numérica', 'Precisión Under/Over Numérica'],
                             title="Precisión por Liga", barmode='group',
                             labels={'value': 'Precisión', 'variable': 'Métrica'})
                fig.update_layout(yaxis_tickformat='.0%')
                st.plotly_chart(fig, use_container_width=True)

    with tab4:
        st.subheader("Predicción Rápida")

        col_pred1, col_pred2, col_pred3 = st.columns(3)

        with col_pred1:
            home_pred = st.selectbox("Equipo Local", all_teams, key='home_pred')
            away_pred = st.selectbox("Equipo Visitante", all_teams, index=1, key='away_pred')

            if st.button("Predecir", key='predict_btn'):
                X_w, X_u = get_prediction_features(df_raw, ppg_dict_full, home_pred, away_pred)
                if X_w is not None:
                    proba_winner = model_winner.predict_proba(X_w)[0]
                    proba_under = model_under.predict_proba(X_u)[0][1]

                    col1, col2, col3 = st.columns(3)
                    col1.metric("Local", f"{proba_winner[0] * 100:.1f}%")
                    col2.metric("Empate", f"{proba_winner[1] * 100:.1f}%")
                    col3.metric("Visitante", f"{proba_winner[2] * 100:.1f}%")

                    col_under, col_over = st.columns(2)
                    col_under.metric("Under 2.5", f"{proba_under * 100:.1f}%")
                    col_over.metric("Over 2.5", f"{(1 - proba_under) * 100:.1f}%")