FEATURE_LABELS = {
    "family_history_with_overweight": "family_history_with_overweight (Familiäre Vorbelastung)",
    "FAVC": "FAVC (High-calorie food häufig)",
    "CAEC": "CAEC (Zwischenmahlzeiten)",
    "SMOKE": "SMOKE (Rauchen)",
    "SCC": "SCC (Kalorien-Tracking)",
    "CALC": "CALC (Alkoholkonsum)",
    "MTRANS": "MTRANS (Transportmittel)",
    "NObeyesdad": "NObeyesdad (Obesity Level)",

    # häufige weitere Kürzel im Obesity-Dataset:
    "FCVC": "FCVC (Gemüsehäufigkeit)",
    "NCP": "NCP (Anzahl Hauptmahlzeiten)",
    "CH2O": "CH2O (Wasser/Tag)",
    "FAF": "FAF (Sporthäufigkeit)",
    "TUE": "TUE (Screen-Time)",
}

def label_col(col: str) -> str:
    """Gibt einen schöneren Anzeigenamen für eine Spalte zurück."""
    return FEATURE_LABELS.get(col, col)