# ============================================
# pages/1_Daten_Exploration.py
# ============================================
"""
Daten Exploration - Woche 7
Data Cleaning und Inspektion
"""
import streamlit as st
import pandas as pd
import numpy as np

from utils.helpers import FEATURE_LABELS, label_col

st.set_page_config(page_title="Daten Exploration", page_icon="📊", layout="wide")

st.title("📊 Daten-Exploration")
st.markdown("Woche 7: Data Cleaning und Inspektion")
st.markdown("---")

DATA_PATH = "data/Final_Data_cleaned.csv"

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Nur für Anzeige: Kürzel -> 'Kürzel (Bedeutung)'."""
    return df.rename(columns=FEATURE_LABELS)

try:
    df_raw = load_data()
    st.success(f"✅ Daten geladen: {len(df_raw)} Zeilen, {len(df_raw.columns)} Spalten")

    # ------------------------------------------------------------
    # Sidebar Filter
    # ------------------------------------------------------------
    st.sidebar.header("Filter Optionen")
    df = df_raw.copy()

    # Age Filter
    if "Age" in df.columns and pd.api.types.is_numeric_dtype(df["Age"]):
        age_min = int(df["Age"].min())
        age_max = int(df["Age"].max())
        age_range = st.sidebar.slider("Alter (Age):", age_min, age_max, (age_min, age_max))
        df = df[(df["Age"] >= age_range[0]) & (df["Age"] <= age_range[1])]

    # Gender Filter
    if "Gender" in df.columns:
        gender_values = sorted(df_raw["Gender"].dropna().unique().tolist())
        selected_genders = st.sidebar.multiselect("Gender (Geschlecht):", gender_values, default=gender_values)
        df = df[df["Gender"].isin(selected_genders)]

    # Obesity Level Filter
    if "NObeyesdad" in df.columns:
        levels = sorted(df_raw["NObeyesdad"].dropna().unique().tolist())
        selected_levels = st.sidebar.multiselect(label_col("NObeyesdad") + ":", levels, default=levels)
        df = df[df["NObeyesdad"].isin(selected_levels)]

    st.sidebar.markdown("---")
    st.sidebar.metric("Gefilterte Zeilen", len(df))
    st.sidebar.metric("Spalten", df.shape[1])

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📋 Übersicht", "🔍 Datenqualität", "📊 Statistiken"])

    # ------------------------------------------------------------
    # Tab 1: Übersicht
    # ------------------------------------------------------------
    with tab1:
        st.subheader("Dataset Übersicht")

        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("#### Daten-Vorschau")
            n_rows = st.slider("Anzahl Zeilen", 5, 50, 10)
            st.dataframe(rename_for_display(df).head(n_rows), use_container_width=True)

        with col2:
            st.markdown("#### Info")
            st.write(f"**Shape (gefiltert):** {df.shape}")
            st.write(f"**Spalten:** {df.shape[1]}")

            info_df = pd.DataFrame({
                "Spalte": [label_col(c) for c in df.columns],
                "Typ": df.dtypes.astype(str).values,
                "Count": df.count().values
            })
            st.markdown("**Datentypen & Non-Null Count:**")
            st.dataframe(info_df, use_container_width=True)

            # Optional: BMI Quick-Check
            if "Height" in df.columns and "Weight" in df.columns:
                try:
                    bmi = df["Weight"] / (df["Height"] ** 2)
                    st.metric("Ø BMI (gefiltert)", f"{bmi.mean():.1f}")
                except Exception:
                    pass

    # ------------------------------------------------------------
    # Tab 2: Datenqualität
    # ------------------------------------------------------------
    with tab2:
        st.subheader("Datenqualität")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("#### Fehlende Werte")
            missing = df_raw.isnull().sum().sort_values(ascending=False)
            if missing.sum() == 0:
                st.success("✅ Keine fehlenden Werte!")
            else:
                miss_df = pd.DataFrame({
                    "Spalte": [label_col(c) for c in missing.index],
                    "Fehlend": missing.values
                })
                miss_df = miss_df[miss_df["Fehlend"] > 0]
                st.dataframe(miss_df, use_container_width=True)

        with col2:
            st.markdown("#### Duplikate")
            n_duplicates = df_raw.duplicated().sum()
            if n_duplicates == 0:
                st.success("✅ Keine Duplikate!")
            else:
                st.warning(f"⚠️ {n_duplicates} Duplikate gefunden")

    # ------------------------------------------------------------
    # Tab 3: Statistiken
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Statistische Zusammenfassung")

        st.markdown("#### Numerische Variablen")
        desc = df.describe()
        desc_display = desc.copy()
        desc_display.columns = [label_col(c) for c in desc.columns]
        st.dataframe(desc_display, use_container_width=True)

        st.markdown("#### Kategorische / Bool Variablen")
        cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns
        if len(cat_cols) > 0:
            for col in cat_cols:
                with st.expander(f"📊 {label_col(col)}"):
                    st.write(df[col].value_counts(dropna=False))

        st.markdown("---")
        st.markdown("#### Wertebereiche (Numerische Spalten)")
        num_cols = df_raw.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) > 0:
            ranges = []
            for c in num_cols:
                ranges.append([label_col(c), df_raw[c].min(), df_raw[c].max(), df_raw[c].mean()])
            ranges_df = pd.DataFrame(ranges, columns=["Spalte", "Min", "Max", "Mean"])
            st.dataframe(ranges_df, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ Datei nicht gefunden: `{DATA_PATH}`")
    st.info("Bitte lege deine bereinigte CSV in den `data/` Ordner und passe ggf. `DATA_PATH` an.")
