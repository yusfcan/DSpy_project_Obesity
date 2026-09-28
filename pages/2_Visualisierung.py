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

# NObeyesdad ist ordinal - alphabetisch sortiert ergibt die Reihenfolge keinen Sinn.
OBESITY_ORDER = [
    "Insufficient_Weight",
    "Normal_Weight",
    "Overweight_Level_I",
    "Overweight_Level_II",
    "Obesity_Type_I",
    "Obesity_Type_II",
    "Obesity_Type_III",
]


# Validierte Farbslots (dark surface) - Reihenfolge ist fix, wird nie zyklisch wiederholt.
SERIES = ["#3987e5", "#d95926", "#199e70", "#c98500", "#d55181", "#008300", "#9085e9"]


def is_ordinal_scale(s: pd.Series) -> bool:
    """Umfrage-Skala (FCVC, NCP, ...) statt echter Messwert?

    Die Rohdaten sind SMOTE-interpoliert, deshalb hat FCVC 810 verschiedene Werte
    obwohl die Frage nur 1/2/3 zuliess. Erkennung: ein nennenswerter Teil der Werte
    liegt exakt auf ganzen Zahlen UND gerundet bleiben wenige Stufen uebrig.
    """
    s = s.dropna()
    if s.empty:
        return False
    return (s % 1 == 0).mean() >= 0.25 and s.round().nunique() <= 8


def order_groups(col: str, values) -> list:
    """Sinnvolle Reihenfolge der Gruppen (ordinal wo möglich, sonst alphabetisch)."""
    values = list(values)
    if col == "NObeyesdad":
        known = [v for v in OBESITY_ORDER if v in values]
        return known + sorted(v for v in values if v not in OBESITY_ORDER)
    return sorted(values)

@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

def rename_for_display(df: pd.DataFrame) -> pd.DataFrame:
    """Nur für Anzeige: Kürzel -> 'Kürzel (Bedeutung)'."""
    return df.rename(columns=FEATURE_LABELS)





try:
    df_raw = load_data()
    st.success(f"✅ Daten geladen: {len(df_raw)} Zeilen, {len(df_raw.columns)} Spalten")
    df_raw["_BMI"] = df_raw["Weight"] / (df_raw["Height"] ** 2)

    # Gender ist im bereinigten CSV 0/1-codiert. Als Zahl landet es in numeric_cols
    # und Plotly macht daraus eine kontinuierliche Farbskala statt zwei Farben.
    # (Zuordnung aus den Daten: Gruppe 0 hat Ø 1.76 m, Gruppe 1 Ø 1.64 m.)
    if "Gender" in df_raw.columns and pd.api.types.is_numeric_dtype(df_raw["Gender"]):
        df_raw["Gender"] = df_raw["Gender"].map({0: "Male", 1: "Female"})

    # Quick Stats (wie beim Dozent)
    c1, c2, c3 = st.columns(3)
    c1.metric("Teilnehmende", len(df_raw))
    c2.metric("Features", df_raw.shape[1])
    c3.metric("Ø BMI", f"{df_raw['_BMI'].mean():.1f}")

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
            hist_mode = "Gestapelt"
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

                    hist_mode = st.radio(
                        "Darstellung:",
                        options=["Gestapelt", "Überlagert (Linien)"],
                        help="Gestapelt zeigt die Zusammensetzung, Linien vergleichen die Formen.",
                    )

        with col2:
            fig, ax = plt.subplots(figsize=(10, 5))

            if split and group_col is not None:
                # Gemeinsame Bin-Kanten aus der Gesamtverteilung: sonst rechnet jede
                # Gruppe eigene Kanten aus ihrem eigenen Min/Max und die Balken haben
                # unterschiedliche Breiten -> die Gruppen sind nicht vergleichbar.
                edges = np.histogram_bin_edges(df[selected_feature].dropna(), bins=bins)

                keys = df[group_col].astype(str)
                groups = order_groups(group_col, keys.unique())
                data = [df.loc[keys == g, selected_feature].dropna() for g in groups]

                if hist_mode == "Gestapelt":
                    ax.hist(data, bins=edges, stacked=True, label=groups,
                            edgecolor="white", linewidth=0.3)
                else:
                    ax.hist(data, bins=edges, histtype="step", label=groups, linewidth=1.8)

                ax.legend(title=label_col(group_col), fontsize=8)
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
            st.caption(
                "Ohne `_BMI`: das ist aus Weight und Height berechnet, "
                "die Korrelation (0.93) wäre also trivial."
            )
            drop = [c for c in ["Weight", "_BMI"] if c in corr.columns]
            target_corr = corr["Weight"].drop(labels=drop).abs().sort_values(ascending=False).head(10)
            show = pd.DataFrame({
                "Feature": [label_col(c) for c in target_corr.index],
                "|corr|": target_corr.values
            })
            st.dataframe(show, use_container_width=True)

    # ------------------------------------------------------------
    # Tab 3: Vergleiche (Diagrammtyp je nach Variablentyp)
    # ------------------------------------------------------------
    with tab3:
        st.subheader("Feature-Vergleiche")

        if len(numeric_cols) < 2:
            st.warning("Zu wenige numerische Spalten für Scatterplots.")
            st.stop()

        # Spalten nach Typ trennen - davon haengt ab, welche Darstellung passt.
        ordinal_cols = [c for c in numeric_cols if is_ordinal_scale(df[c])]
        continuous_cols = [c for c in numeric_cols if c not in ordinal_cols]

        def var_kind(col: str) -> str:
            if col in cat_cols:
                return "kategorial"
            return "ordinal" if col in ordinal_cols else "kontinuierlich"

        axis_options = continuous_cols + ordinal_cols + cat_cols

        c1, c2 = st.columns(2)
        with c1:
            x_feature = st.selectbox(
                "X-Achse:", axis_options,
                index=axis_options.index("NObeyesdad") if "NObeyesdad" in axis_options else 0,
                format_func=label_col,
            )
        with c2:
            y_opts = continuous_cols or numeric_cols
            y_feature = st.selectbox(
                "Y-Achse (Messwert):", y_opts,
                index=y_opts.index("Weight") if "Weight" in y_opts else 0,
                format_func=label_col,
            )

        # Empfehlung aus den Variablentypen ableiten
        xk, yk = var_kind(x_feature), var_kind(y_feature)
        if xk == "kontinuierlich" and yk == "kontinuierlich":
            empfohlen, grund = "Scatter", "beide Achsen sind echte Messwerte"
        elif xk == "kontinuierlich" or yk == "kontinuierlich":
            diskret = x_feature if xk != "kontinuierlich" else y_feature
            empfohlen = "Box-Plot"
            grund = f"`{label_col(diskret)}` ist {var_kind(diskret)} - ein Scatter legt die Punkte auf wenige Linien"
        else:
            empfohlen, grund = "Dichte-Heatmap", "beide Achsen sind diskret - Punkte wuerden exakt uebereinander liegen"

        typen = ["Box-Plot", "Violin", "Scatter", "Dichte-Heatmap"]
        chart_type = st.radio(
            "Diagrammtyp:", typen, index=typen.index(empfohlen),
            horizontal=True,
            help="Vorausgewaehlt ist der Typ, der zu den Variablentypen passt.",
        )
        st.caption(f"Empfohlen: **{empfohlen}** - {grund}.")

        labels = {col: label_col(col) for col in df.columns}
        plot_df = df.copy()

        # Ordinale Achsen runden: FCVC hat 810 SMOTE-Werte, ungerundet gaebe das 810 Boxen.
        def as_group_axis(col: str) -> str:
            if col in ordinal_cols:
                key = f"{col}_stufe"
                plot_df[key] = plot_df[col].round().astype(int)
                labels[key] = f"{label_col(col)} - gerundet"
                return key
            return col

        if chart_type in ("Box-Plot", "Violin"):
            gx = as_group_axis(x_feature)
            order = order_groups(x_feature, plot_df[gx].astype(str).unique())
            plot_df[gx] = plot_df[gx].astype(str)

            common = dict(
                x=gx, y=y_feature, labels=labels,
                category_orders={gx: order},
                color_discrete_sequence=[SERIES[0]],
                title=f"{label_col(y_feature)} nach {label_col(x_feature)}",
            )
            # Identitaet traegt die X-Achse, nicht die Farbe -> eine Farbe, keine Legende.
            fig = (px.box(plot_df, points="outliers", **common) if chart_type == "Box-Plot"
                   else px.violin(plot_df, box=True, **common))
            # Duenne Marks: transparente Flaeche, praezise Kontur - keine satten Bloecke.
            fig.update_traces(
                fillcolor="rgba(57, 135, 229, 0.22)",
                line=dict(color=SERIES[0], width=2),
                marker=dict(size=4, opacity=0.5),
                width=0.55,
            )
            fig.update_xaxes(type="category")

        elif chart_type == "Scatter":
            color_options = [None] + [c for c in cat_cols]
            color_by = st.selectbox(
                "Färben nach (optional):", color_options,
                format_func=lambda v: "Keine" if v is None else label_col(v),
            )
            facet = None
            if color_by is not None:
                n = plot_df[color_by].nunique()
                # Im Scatter muss jede Farbe von jeder anderen unterscheidbar sein
                # (alle Paare, nicht nur benachbarte) - darueber wird facettiert.
                if n > 3:
                    st.info(
                        f"`{label_col(color_by)}` hat {n} Auspraegungen. Im Scatter sind mehr als "
                        "3 Farben nicht mehr sicher trennbar - deshalb als kleine Vielfache nebeneinander."
                    )
                    facet, color_by = color_by, None

            key = facet or color_by
            orders = {key: order_groups(key, plot_df[key].astype(str).unique())} if key else {}
            # OLS braucht zwei numerische Achsen - bei kategorialem X gibt es keine Gerade.
            trend = "ols" if x_feature in numeric_cols else None
            fig = px.scatter(
                plot_df, x=x_feature, y=y_feature, color=color_by, facet_col=facet,
                facet_col_wrap=4, labels=labels, category_orders=orders, opacity=0.5,
                color_discrete_sequence=SERIES,
                trendline=trend, trendline_color_override="#c3c2b7",
                title=f"{label_col(y_feature)} vs {label_col(x_feature)}",
            )
            fig.update_traces(marker=dict(size=6))

        else:  # Dichte-Heatmap
            fig = px.density_heatmap(
                plot_df, x=as_group_axis(x_feature), y=y_feature, labels=labels,
                nbinsx=None if x_feature in cat_cols else 25, nbinsy=25,
                color_continuous_scale="Blues",
                title=f"Häufigkeitsdichte: {label_col(y_feature)} vs {label_col(x_feature)}",
            )

        fig.update_layout(margin=dict(t=60, b=40))
        st.plotly_chart(fig, use_container_width=True)

        # Jede Grafik braucht eine Tabellen-Entsprechung (Werte nie nur ueber Farbe/Hover).
        with st.expander("📋 Werte als Tabelle"):
            if chart_type in ("Box-Plot", "Violin"):
                gx = as_group_axis(x_feature)
                stats = (df.assign(_g=plot_df[gx].astype(str))
                           .groupby("_g")[y_feature]
                           .agg(n="count", Median="median", Mittelwert="mean",
                                Q1=lambda s: s.quantile(.25), Q3=lambda s: s.quantile(.75))
                           .reindex(order_groups(x_feature, plot_df[gx].astype(str).unique())))
                st.dataframe(stats.round(2), use_container_width=True)
            else:
                st.dataframe(
                    df[[x_feature, y_feature]].describe().T.round(2),
                    use_container_width=True,
                )

except FileNotFoundError:
    st.error(f"❌ Datei nicht gefunden: `{DATA_PATH}`")
    st.info("Bitte lege deine bereinigte CSV in den `data/` Ordner und passe ggf. `DATA_PATH` an.")
