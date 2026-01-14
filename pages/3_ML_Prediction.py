# ============================================
# pages/3_ML_Prediction.py
# ============================================
"""
ML Prediction - Woche 9
Regression: Weight vorhersagen
(Notebook-Code -> Streamlit)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt  # <-- NEU

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

st.set_page_config(page_title="ML Prediction", page_icon="🤖", layout="wide")

st.title("🤖 Machine Learning Prediction")
st.markdown("Woche 9: **Regression – Weight vorhersagen**")
st.markdown("---")

DATA_PATH = "data/Final_Data_cleaned.csv"

# Genau wie in deinem Notebook:
features = ["Height", "Age", "FAF", "family_history_with_overweight", "Gender"]
target = "Weight"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def get_model(model_choice: str):
    if model_choice == "Linear Regression":
        return LinearRegression()
    if model_choice == "K-Nearest Neighbors Regressor":
        return KNeighborsRegressor(n_neighbors=5)
    if model_choice == "Decision Tree Regressor":
        return DecisionTreeRegressor(max_depth=5, random_state=42)
    # Random Forest Regressor (Default)
    return RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)

@st.cache_data
def compute_model_comparison(df_model: pd.DataFrame, test_size: float):
    X = df_model[features].copy()
    y = df_model[target].copy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )

    models = {
        "Linear Regression": LinearRegression(),
        "KNN Regressor": KNeighborsRegressor(n_neighbors=5),
        "Decision Tree": DecisionTreeRegressor(max_depth=5, random_state=42),
        "Random Forest": RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1),
    }

    rows = []
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        mse = mean_squared_error(y_test, y_pred)
        rmse = float(np.sqrt(mse))
        r2 = float(r2_score(y_test, y_pred))

        rows.append({"Model": name, "RMSE": rmse, "R² Score": r2})

    results = pd.DataFrame(rows).sort_values("R² Score", ascending=False)
    return results

try:
    df = load_data()
    st.success(f"✅ Daten geladen: {len(df)} Zeilen, {len(df.columns)} Spalten")

    # Check: Spalten vorhanden?
    missing_cols = [c for c in features + [target] if c not in df.columns]
    if missing_cols:
        st.error(f"❌ Diese Spalten fehlen in deiner CSV: {missing_cols}")
        st.stop()

    # Tabs wie beim Dozent
    tab1, tab2 = st.tabs(["🎯 Training & Evaluation", "🔮 Vorhersage"])

    # ------------------------------------------
    # Tab 1: Training & Evaluation
    # ------------------------------------------
    with tab1:
        st.subheader("Model Training")

        # Notebook: df_model = df[features + [target]].dropna()
        df_model = df[features + [target]].dropna()
        X = df_model[features].copy()
        y = df_model[target].copy()

        c1, c2, c3 = st.columns(3)
        c1.metric("Anzahl Samples", len(X))
        c2.metric("Anzahl Features", X.shape[1])
        c3.metric("Ø Weight", f"{y.mean():.2f}")

        left, right = st.columns([1, 2])

        with left:
            st.markdown("#### Einstellungen")

            model_choice = st.selectbox(
                "Modell:",
                [
                    "Random Forest Regressor",
                    "Linear Regression",
                    "K-Nearest Neighbors Regressor",
                    "Decision Tree Regressor",
                ],
            )

            test_size = st.slider("Test-Set Größe:", 0.1, 0.4, 0.2, 0.05)

            train_button = st.button("🚀 Training starten", type="primary")

        with right:
            # ✅ Modellvergleich immer sichtbar (ohne Button)
            st.markdown("#### 📊 Modellvergleich (Orientierung)")
            results = compute_model_comparison(df_model, test_size)
            st.dataframe(results, use_container_width=True)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

            ax1.bar(results["Model"], results["RMSE"])
            ax1.set_title("Modellvergleich - RMSE (niedriger = besser)")
            ax1.set_ylabel("RMSE")
            ax1.tick_params(axis="x", rotation=45)
            ax1.grid(axis="y", alpha=0.3)

            ax2.bar(results["Model"], results["R² Score"])
            ax2.set_title("Modellvergleich - R² Score (höher = besser)")
            ax2.set_ylabel("R² Score")
            ax2.set_ylim(0, 1)
            ax2.tick_params(axis="x", rotation=45)
            ax2.grid(axis="y", alpha=0.3)
            for i, v in enumerate(results["R² Score"].values):
                ax2.text(i, v + 0.02, f"{v:.2%}", ha="center")

            plt.tight_layout()
            st.pyplot(fig)

            # Dein bestehender Training-Flow bleibt gleich
            if train_button:
                with st.spinner("Trainiere Modell..."):
                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, random_state=42
                    )

                    model = get_model(model_choice)
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)

                    mse = mean_squared_error(y_test, y_pred)
                    rmse = np.sqrt(mse)
                    mae = mean_absolute_error(y_test, y_pred)
                    r2 = r2_score(y_test, y_pred)

                    st.session_state["best_model"] = model
                    st.session_state["features"] = features

                    st.success("✅ Training abgeschlossen!")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("MSE", f"{mse:.2f}")
                    m2.metric("RMSE", f"{rmse:.2f}")
                    m3.metric("MAE", f"{mae:.2f}")
                    m4.metric("R²", f"{r2:.2%}")

                    st.markdown("#### Test-Set Übersicht")
                    st.write(f"Training Set: {len(X_train)} samples")
                    st.write(f"Test Set: {len(X_test)} samples")

                    with st.expander("📋 Target-Statistik (Weight)"):
                        st.write(y.describe())

    # ------------------------------------------
    # Tab 2: Vorhersage
    # ------------------------------------------
    with tab2:
        st.subheader("Einzelne Vorhersage (Weight)")
        st.info("💡 Erst im Training-Tab ein Modell trainieren, dann hier Werte eingeben.")

        if "best_model" not in st.session_state:
            st.warning("⚠️ Kein trainiertes Modell gefunden. Bitte zuerst Training starten.")
            st.stop()

        best_model = st.session_state["best_model"]

        with st.form("prediction_form"):
            st.markdown("#### Eingabedaten")

            col1, col2, col3 = st.columns(3)

            with col1:
                height = st.number_input("Height (m)", min_value=0.5, max_value=2.5, value=1.83, step=0.01)
                age = st.number_input("Age", min_value=5, max_value=120, value=25, step=1)

            with col2:
                faf = st.number_input("FAF", min_value=0.0, max_value=10.0, value=1.0, step=0.1)
                fam_hist = st.selectbox("family_history_with_overweight", options=[False, True], index=0)

            with col3:
                gender = st.selectbox(
                    "Gender",
                    options=[0, 1],
                    format_func=lambda x: "Männlich (0)" if x == 0 else "Weiblich (1)"
                )

            submitted = st.form_submit_button("🔮 Vorhersage", type="primary")

        if submitted:
            new_data_raw = pd.DataFrame({
                "Height": [height],
                "Age": [age],
                "FAF": [faf],
                "family_history_with_overweight": [fam_hist],
                "Gender": [gender],
            })

            prediction = best_model.predict(new_data_raw)
            st.success(f"✅ Vorhersage für Weight: **{prediction[0]:.2f}**")

except FileNotFoundError:
    st.error(f"❌ Datei nicht gefunden: `{DATA_PATH}`")
    st.info("Bitte lege deine bereinigte CSV in den `data/` Ordner und passe ggf. `DATA_PATH` an.")

