"""
Funciones de preparación de datos para el dataset Titanic.
- Carga datos desde datos/datos_ini/titanic.csv
- Limpia valores faltantes
- Realiza codificación básica
- Guarda datos limpios en datos/datos_limp/titanic_clean.csv
"""
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATOS_INI = ROOT / "datos" / "datos_ini" / "titanic.csv"
DATOS_LIMP_DIR = ROOT / "datos" / "datos_limp"
DATOS_LIMP_DIR.mkdir(parents=True, exist_ok=True)


def load_data(path: str = None):
    p = DATOS_INI if path is None else Path(path)
    df = pd.read_csv(p)
    return df


def preprocess(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop columns with many identifiers/low value
    df = df.drop([c for c in ["PassengerId", "Name", "Ticket", "Cabin"] if c in df.columns], axis=1)
    # Fill Age with median
    if "Age" in df.columns:
        df["Age"] = df["Age"].fillna(df["Age"].median())
    # Fill Embarked with mode
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Fill Fare
    if "Fare" in df.columns:
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())
    # Encode Sex
    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].map({"male": 0, "female": 1}).astype(int)
    # One-hot Embarked
    if "Embarked" in df.columns:
        df = pd.get_dummies(df, columns=["Embarked"], prefix="Embarked", drop_first=True)
    # Ensure no NA
    df = df.dropna()
    return df


def save_clean(df: pd.DataFrame, path: str = None):
    target = DATOS_LIMP_DIR / "titanic_clean.csv" if path is None else Path(path)
    df.to_csv(target, index=False)
    print(f"Datos limpios guardados en {target}")
    return target


if __name__ == "__main__":
    df = load_data()
    dfc = preprocess(df)
    save_clean(dfc)
