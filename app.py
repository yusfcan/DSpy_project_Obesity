# ============================================
# app.py (HOME PAGE)
# ============================================
"""
Obesity / Weight Prediction App
Hauptseite / Home
"""
import streamlit as st
import pandas as pd
import numpy as np

# Page Config
st.set_page_config(
    page_title="Obesity Weight App",
    page_icon="⚖️",
    layout="wide"
)

st.title("⚖️ Obesity / Weight Prediction App")
st.markdown("### Datenanalyse und ML-Vorhersage (Regression: Weight)")

st.markdown("---")

st.markdown("""
## Willkommen! 👋

Diese App analysiert ein **Obesity-Dataset** mit **Menschen unterschiedlicher Altersgruppen, Geschlechter und Lebensgewohnheiten**
(z.B. Ernährungs- und Aktivitätsverhalten) und erstellt eine **ML-Vorhersage für das Gewicht (Weight)**.

### 📚 Bereiche:
1. **📊 Daten-Exploration** – Datensatz inspizieren & Datenqualität prüfen  
2. **📈 Visualisierung** – Explorative Datenanalyse (EDA)  
3. **🤖 ML Prediction (Regression)** – **Weight vorhersagen** (Training & Prediction)

👈 **Nutzen Sie die Sidebar**, um zwischen den Seiten zu navigieren!
""")

st.markdown("---")
st.markdown("### 📊 Dataset Info")

DATA_PATH = "data/Final_Data_cleaned.csv"

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    return pd.read_csv(path)

def safe_mean(df: pd.DataFrame, col: str) -> float:
    return float(df[col].mean()) if col in df.columns else 0.0

try:
    df = load_data(DATA_PATH)

    # Optional: BMI berechnen (falls Height in Metern vorliegt)
    if "Height" in df.columns and "Weight" in df.columns:


        df = df.replace([np.inf, -np.inf], np.nan)
        dataset_info = df.describe()

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Teilnehmende", len(df))

    with col2:
        st.metric("Features", df.shape[1])

    with col3:
        st.metric("Ø Alter", f"{safe_mean(df, 'Age'):.1f}")

    with col4:
        st.metric("Ø Gewicht", f"{safe_mean(df, 'Weight'):.1f} kg")

    # Zusatzinfo unter den Metrics (kurz & clean)
    extras = []
    if "NObeyesdad" in df.columns:
        extras.append(f"Obesity-Level Klassen: **{df['NObeyesdad'].nunique()}**")
    if "_BMI" in df.columns:
        extras.append(f"Ø BMI: **{df['_BMI'].mean():.1f}**")

    if extras:
        st.info(" · ".join(extras))
    st.markdown("---")
    st.markdown("### 📋 Daten-Vorschau")
    st.dataframe(df.head(10))

except FileNotFoundError:
    st.warning(f"⚠️ Datei nicht gefunden: `{DATA_PATH}`")
    st.info("Bitte legen Sie die bereinigte CSV-Datei in den `data/` Ordner (Cleaning-Code kommt nicht in die App).")

st.markdown("---")
st.caption("DataPy WiSe25/26 – Obesity / Weight Prediction Project")
