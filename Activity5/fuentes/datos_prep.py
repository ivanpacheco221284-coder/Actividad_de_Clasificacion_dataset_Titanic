"""
datos_prep.py
Funciones para carga y limpieza del dataset Titanic
"""

import pandas as pd
from sklearn.model_selection import train_test_split


def load_raw(path: str) -> pd.DataFrame:
    """Carga el CSV original."""
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza básica: selecciona columnas útiles, rellena nulos y codifica."""
    df = df.copy()
    # Select useful columns
    keep = ["Survived","Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Name"]
    df = df[keep]
    # Fill Age with median
    df["Age"] = df["Age"].fillna(df["Age"].median())
    # Fill Embarked with mode
    df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])
    # Feature engineering: title
    df['Title'] = df['Name'].str.extract(r',\s*([^\.]+)\.')[0]
    df['Title'] = df['Title'].replace(['Lady','Countess','Capt','Col','Don','Dr','Major','Rev','Sir','Jonkheer','Dona'],'Rare')
    df['Title'] = df['Title'].replace('Mlle','Miss')
    df['Title'] = df['Title'].replace('Ms','Miss')
    df['Title'] = df['Title'].replace('Mme','Mrs')
    df = df.drop(columns=['Name'])
    # One-hot encoding for categorical features
    df = pd.get_dummies(df, columns=["Sex","Embarked","Title"], drop_first=True)
    return df


def save_clean(df: pd.DataFrame, path: str):
    df.to_csv(path, index=False)


def prepare_for_model(path_raw: str, path_clean: str) -> pd.DataFrame:
    df = load_raw(path_raw)
    df_clean = basic_clean(df)
    save_clean(df_clean, path_clean)
    return df_clean


if __name__ == "__main__":
    import sys
    raw = sys.argv[1] if len(sys.argv)>1 else 'Activity5/datos/datos_ini/titanic.csv'
    clean = sys.argv[2] if len(sys.argv)>2 else 'Activity5/datos/datos_limp/titanic_clean.csv'
    dfc = prepare_for_model(raw, clean)
    print('Saved cleaned data to', clean)
