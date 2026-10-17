"""
Funciones para la preparación de datos del dataset Titanic.
Guarda archivos limpios en datos/datos_limp/titanic_clean.csv
"""
import os
import pandas as pd
import numpy as np


def load_raw(path='datos/datos_ini/titanic.csv'):
    return pd.read_csv(path)


def basic_clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # Drop columns not useful for modeling
    drop_cols = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)

    # Fill Embarked with mode
    if 'Embarked' in df.columns:
        df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Age: impute with median
    if 'Age' in df.columns:
        df['Age'].fillna(df['Age'].median(), inplace=True)

    # Fare: fill with median
    if 'Fare' in df.columns:
        df['Fare'].fillna(df['Fare'].median(), inplace=True)

    # Transform Sex to numeric
    if 'Sex' in df.columns:
        df['Sex'] = df['Sex'].map({'male': 0, 'female': 1}).astype(int)

    # One-hot encode Embarked
    if 'Embarked' in df.columns:
        df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    # Create family size feature
    if all(col in df.columns for col in ['SibSp', 'Parch']):
        df['FamilySize'] = df['SibSp'] + df['Parch']

    # Ensure no remaining missing
    # For safety, fill any remaining numeric NaNs with median
    for c in df.select_dtypes(include=[np.number]).columns:
        if df[c].isna().any():
            df[c].fillna(df[c].median(), inplace=True)

    return df


def save_clean(df: pd.DataFrame, dest='datos/datos_limp/titanic_clean.csv'):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    df.to_csv(dest, index=False)
    print(f'Saved cleaned data to {dest}')


if __name__ == '__main__':
    df = load_raw()
    dfc = basic_clean(df)
    save_clean(dfc)
