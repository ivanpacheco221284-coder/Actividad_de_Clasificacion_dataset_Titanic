import os
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent.parent
DATASET_PATH = ROOT_DIR / "datos" / "datos_ini" / "titanic.csv"
CLEAN_DATA_PATH = ROOT_DIR / "datos" / "datos_limp" / "titanic_limpio.csv"


def ensure_dataset(path: Path = DATASET_PATH) -> Path:
    """Valida que exista el archivo del dataset Titanic; si no, intenta cargarlo del paquete seaborn."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size > 0:
        return path

    try:
        import seaborn as sns

        df = sns.load_dataset("titanic")
        df.to_csv(path, index=False)
        return path
    except Exception as exc:  # pragma: no cover
        raise FileNotFoundError(
            f"No se encontró el dataset Titanic en {path}. "
            "Instala seaborn o añade el archivo titanic.csv." 
        ) from exc


def load_dataset(path: Path | None = None) -> pd.DataFrame:
    """Carga el dataset Titanic desde disco o lo genera si es necesario."""
    dataset_path = path or ensure_dataset()
    return pd.read_csv(dataset_path)


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpieza básica y preparación de atributos para entrenamiento."""
    data = df.copy()

    # Columnas de identificacion o irrelevantes para clasificacion
    drop_cols = ["PassengerId", "Name", "Ticket", "Cabin"]
    data = data.drop(columns=[col for col in drop_cols if col in data.columns], errors="ignore")

    # Mapeo de la variable objetivo
    if "Survived" in data.columns:
        data = data.rename(columns={"Survived": "survived"})

    # Normalizacion de columnas 
    if "Sex" in data.columns:
        data["Sex"] = data["Sex"].astype(str).str.lower()
    if "Embarked" in data.columns:
        data["Embarked"] = data["Embarked"].astype(str).str.strip().str.upper()

    # Imputacion numerica
    for col in ["Age", "Fare"]:
        if col in data.columns:
            median_value = data[col].median()
            data[col] = data[col].fillna(median_value)

    # Imputacion categorica
    for col in ["Embarked", "Sex"]:
        if col in data.columns:
            mode_value = data[col].mode(dropna=True)
            if not mode_value.empty:
                data[col] = data[col].fillna(mode_value.iloc[0])

    # Asegurar valores categoricos para evitar problemas en preprocesamiento
    for col in data.columns:
        if data[col].dtype == "object":
            data[col] = data[col].fillna("desconocido")

    return data


def save_clean_dataset(df: pd.DataFrame, output_path: Path = CLEAN_DATA_PATH) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_preprocessor() -> object:
    """Preprocesador con imputacion, one-hot encoding y escalado para columnas numericas."""
    from sklearn.compose import ColumnTransformer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler
    from sklearn.impute import SimpleImputer

    numeric_features = ["Age", "SibSp", "Parch", "Fare", "Pclass"]
    categorical_features = ["Sex", "Embarked"]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore"))
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features)
        ]
    )

    return preprocessor


def prepare_features_for_training(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Retorna X e y listos para entrenamiento."""
    data = clean_dataset(df)
    if "survived" not in data.columns:
        raise KeyError("La columna 'survived' no existe en el dataset.")

    target = data["survived"].astype(int)
    features = data.drop(columns=["survived"])

    # Reordenar columnas para mantener consistencia del modelo
    features = features[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked"]]
    return features, target
