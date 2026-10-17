"""Funciones de preparación de datos para el dataset Titanic.

Funciones:
- download_dataset(dest_path): descarga el CSV si no existe.
- load_data(path): lee el CSV y devuelve DataFrame.
- preprocess(df, training=True): limpia y realiza transformaciones básicas y devuelve X, y (si training)
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

URL = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"


def download_dataset(dest_path="datos/datos_ini/titanic.csv"):
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    if not os.path.exists(dest_path):
        print(f"Descargando dataset desde {URL} a {dest_path} ...")
        df = pd.read_csv(URL)
        df.to_csv(dest_path, index=False)
        print("Descarga completada.")
    else:
        print(f"Dataset ya existe en {dest_path}")
    return dest_path


def load_data(path="datos/datos_ini/titanic.csv"):
    if not os.path.exists(path):
        download_dataset(path)
    return pd.read_csv(path)


def preprocess(df, training=True, random_state=42):
    """Limpieza y transformación básica.
    - Selección de columnas relevantes
    - Relleno de nulos
    - Encoding básico para Sex y Embarked
    - Creación de feature AgeBin (opcional)
    - Devuelve X,y si training else X
    """
    df = df.copy()
    # Keep a subset of predictive columns
    cols = ["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
    if training:
        cols = ["Survived"] + cols
    df = df[cols] if training else df[cols]

    # Embarked: fill with mode
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].mode().iloc[0])

    # Age: fill with median
    df['Age'] = df['Age'].fillna(df['Age'].median())

    # Fare: fill with median
    df['Fare'] = df['Fare'].fillna(df['Fare'].median())

    # Sex encoding
    df['Sex'] = df['Sex'].map({'male':0,'female':1}).astype(int)

    # One-hot Embarked
    df = pd.get_dummies(df, columns=['Embarked'], drop_first=True)

    if training:
        y = df['Survived'].astype(int)
        X = df.drop(columns=['Survived'])
        return X, y
    else:
        return df


if __name__ == "__main__":
    p = download_dataset()
    df = load_data(p)
    X,y = preprocess(df)
    print("Datos preparados:", X.shape, y.shape)
