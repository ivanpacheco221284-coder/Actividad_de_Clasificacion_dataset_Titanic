import os
from pathlib import Path

import pandas as pd


def load_raw_data(path: str | os.PathLike) -> pd.DataFrame:
    """Carga el dataset Titanic original."""
    return pd.read_csv(path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza simple, reproducible y documentada."""
    cleaned = df.copy()

    # columnas irrelevantes para el modelo
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    cleaned = cleaned.drop(columns=[c for c in drop_cols if c in cleaned.columns], errors="ignore")

    # imputación numérica
    for col in ["Age", "Fare"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].median())

    # imputación categórica
    for col in ["Embarked", "Sex"]:
        if col in cleaned.columns:
            cleaned[col] = cleaned[col].fillna(cleaned[col].mode()[0])

    # normalización de tipos
    if "Survived" in cleaned.columns:
        cleaned["Survived"] = cleaned["Survived"].astype(int)

    cleaned["Sex"] = cleaned["Sex"].astype(str).str.lower()
    cleaned["Embarked"] = cleaned["Embarked"].astype(str).str.strip().str.upper()

    return cleaned


def save_clean_dataset(df: pd.DataFrame, output_path: str | os.PathLike) -> None:
    """Guarda el dataframe limpio en CSV."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)


if __name__ == "__main__":
    base_dir = Path(__file__).resolve().parents[1]
    raw_path = base_dir / "datos" / "datos_ini" / "titanic.csv"
    clean_path = base_dir / "datos" / "datos_limp" / "titanic_limpio.csv"

    if not raw_path.exists():
        raise FileNotFoundError(f"No se encontró el CSV original en: {raw_path}")

    df = load_raw_data(raw_path)
    cleaned = clean_dataset(df)
    save_clean_dataset(cleaned, clean_path)
    print(f"Datos limpios guardados en: {clean_path}")
    print(cleaned.head())
