# ============================================
# pages/2_Visualisierung.py
# ============================================
"""
Visualisierung - Woche 8
Explorative Datenanalyse (EDA)
(Obesity Dataset)
"""
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

from utils.helpers import FEATURE_LABELS, label_col

st.set_page_config(page_title="Visualisierung", page_icon="📈", layout="wide")

st.title("📈 Datenvisualisierung")
st.markdown("Woche 8: Explorative Datenanalyse (EDA)")
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

    # Quick Stats (wie beim Dozent)
    c1, c2, c3 = st.columns(3)
    c1.metric("Teilnehmende", len(df_raw))
    c2.metric("Features", df_raw.shape[1])

    if "Weight" in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw["Weight"]):
        c3.metric("Ø Weight", f"{df_raw['Weight'].mean():.1f}")
    else:
        c3.metric("Ø Weight", "n/a")

    # Für Tabs arbeiten wir mit df (optional: später Filter)
    df = df_raw.copy()

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()

    # Tabs
    tab1, tab2, tab3 = st.tabs(["📊 Verteilungen", "🔗 Korrelationen", "📉 Vergleiche"])

    # ------------------------------------------------------------
    # Tab 1: Verteilungen
    # ------------------------------------------------------------
    with tab1:
        st.subheader("Verteilungsanalyse")

        col1, col2 = st.columns([1, 3])

        with col1:
            if len(numeric_cols) == 0:
                st.warning("Keine numerischen Spalten gefunden.")
                st.stop()

            selected_feature = st.selectbox(
                "Feature wählen:",
                options=numeric_cols,
                format_func=label_col
            )

            bins = st.slider("Bins (Klassen)", 5, 60, 20)

            split = st.checkbox("Nach Kategorie aufteilen")

            group_col = None
            if split:
                # sinnvolle Gruppierung: bevorzugt NObeyesdad/Gender, falls vorhanden
                preferred = []
                for c in ["NObeyesdad", "Gender"]:
                    if c in df.columns:
                        preferred.append(c)

                options = preferred + [c for c in df.columns if c not in preferred and c in cat_cols]
                # falls z.B. Gender numerisch ist, erlauben wir es trotzdem:
                if "Gender" in df.columns and "Gender" not in options:
                    options = ["Gender"] + options

                if len(options) == 0:
                    st.info("Keine kategorischen Spalten gefunden zum Aufteilen.")
                    split = False
                else:
                    group_col = st.selectbox(
                        "Kategorie wählen:",
                        options=options,
                        format_func=label_col
                    )

        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))

            if split and group_col is not None:
                # max. 6 Gruppen anzeigen (sonst unübersichtlich)
                groups = df[group_col].astype(str).fillna("NaN").unique().tolist()
                groups = sorted(groups)[:6]

                for g in groups:
                    subset = df[df[group_col].astype(str).fillna("NaN") == g]
                    ax.hist(subset[selected_feature].dropna(), bins=bins, alpha=0.5, label=str(g))

                ax.legend(title=label_col(group_col))
                ax.set_title(f"Verteilung: {label_col(selected_feature)} (nach {label_col(group_col)})")
            else:
                ax.hist(df[selected_feature].dropna(), bins=bins, edgecolor="black")
                ax.set_title(f"Verteilung: {label_col(selected_feature)}")

            ax.set_xlabel(label_col(selected_feature))
            ax.set_ylabel("Häufigkeit")
            st.pyplot(fig)

    # ------------------------------------------------------------
    # Tab 2: Korrelationen
    # ------------------------------------------------------------
    with tab2:
        st.subheader("Korrelations-Analyse")

        if len(numeric_cols) < 2:
            st.warning("Zu wenige numerische Spalten für eine Korrelationsmatrix.")
            st.stop()

        numeric_df = df[numeric_cols]
        corr = numeric_df.corr()

        # Interaktive Heatmap mit Plotly (ohne seaborn)
        fig = px.imshow(
            corr,
            text_auto=".2f",
            aspect="auto",
            title="Korrelationsmatrix (numerische Features)"
        )
        st.plotly_chart(fig, use_container_width=True)

        # Top Korrelationen mit Weight (wenn vorhanden)
        if "Weight" in corr.columns:
            st.markdown("#### Top Korrelationen mit Weight")
            target_corr = corr["Weight"].abs().sort_values(ascending=False)[1:11]
            show = pd.DataFrame({
                "Feature": [label_col(c) for c in target_corr.index],
                "|corr|": target_corr.values
            })
            st.dataframe(show, use_container_width=True)

    # ------------------------------------------------------------
    # Tab 3: Vergleiche (Scatter)
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Feature-Vergleiche")

        if len(numeric_cols) < 2:
            st.warning("Zu wenige numerische Spalten für Scatterplots.")
            st.stop()

        c1, c2, c3 = st.columns(3)

        with c1:
            x_feature = st.selectbox("X-Achse:", numeric_cols, index=0, format_func=label_col)

        with c2:
            y_default = 1 if len(numeric_cols) > 1 else 0
            y_feature = st.selectbox("Y-Achse:", numeric_cols, index=y_default, format_func=label_col)

        with c3:
            color_options = [None]
            # auch NObeyesdad/Gender anbieten, selbst wenn Gender numerisch ist
            for c in ["NObeyesdad", "Gender"]:
                if c in df.columns and c not in color_options:
                    color_options.append(c)
            for c in cat_cols:
                if c not in color_options:
                    color_options.append(c)

            color_by = st.selectbox(
                "Färben nach (optional):",
                options=color_options,
                format_func=lambda v: "Keine" if v is None else label_col(v)
            )

        labels = {col: label_col(col) for col in df.columns}

        fig = px.scatter(
            df,
            x=x_feature,
            y=y_feature,
            color=color_by,
            title=f"{label_col(y_feature)} vs {label_col(x_feature)}",
            labels=labels,
        )
        st.plotly_chart(fig, use_container_width=True)

except FileNotFoundError:
    st.error(f"❌ Datei nicht gefunden: `{DATA_PATH}`")
    st.info("Bitte lege deine bereinigte CSV in den `data/` Ordner und passe ggf. `DATA_PATH` an.")
