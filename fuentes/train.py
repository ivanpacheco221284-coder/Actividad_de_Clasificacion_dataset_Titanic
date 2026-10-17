#!/usr/bin/env python3
"""
train.py
Script para entrenar modelos de clasificación (LogisticRegression y RandomForest) sobre Titanic.
Registra experimentos en MLflow (local mlruns/ por defecto).

Genera:
- parámetros y métricas
- artefactos: modelo, matriz de confusión
"""
import argparse
from pathlib import Path
import joblib
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
CLEAN_PATH = ROOT / "datos" / "datos_limp" / "titanic_clean.csv"


def load_data(path=CLEAN_PATH):
    df = pd.read_csv(path)
    return df


def eval_and_log(estimator, X_test, y_test, run_name, model_name):
    y_pred = estimator.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    mlflow.log_metric('accuracy', acc)
    mlflow.log_metric('precision', prec)
    mlflow.log_metric('recall', rec)
    mlflow.log_metric('f1', f1)

    # save confusion matrix
    plt.figure(figsize=(5,4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True')
    plt.xlabel('Pred')
    fname = f'confusion_matrix_{model_name}.png'
    plt.savefig(fname)
    plt.close()
    mlflow.log_artifact(fname)

    print(f"{model_name} - acc={acc:.4f} f1={f1:.4f}")
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def main(args):
    mlflow.set_experiment(args.experiment)
    df = load_data()
    # target
    if 'Survived' not in df.columns:
        raise ValueError('El dataset no contiene la columna Survived')
    y = df['Survived']
    X = df.drop(columns=['Survived'])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    cv = StratifiedKFold(n_splits=args.k, shuffle=True, random_state=42)

    models = [
        ('logreg', LogisticRegression(solver='liblinear', max_iter=1000),
         {'C': [0.01, 0.1, 1.0, 10.0]}),
        ('rf', RandomForestClassifier(random_state=42),
         {'n_estimators': [50, 100], 'max_depth': [None, 5, 10]})
    ]

    for name, model, param_grid in models:
        with mlflow.start_run(run_name=f"train_{name}"):
            print(f"Entrenando {name} ...")
            grid = GridSearchCV(model, param_grid, cv=cv, scoring='f1', n_jobs=-1)
            grid.fit(X_train, y_train)
            best = grid.best_estimator_
            mlflow.log_params({f'{name}_{k}': v for k, v in grid.best_params_.items()})
            mlflow.log_param('model', name)
            # Log CV best score
            mlflow.log_metric('best_cv_score', grid.best_score_)

            # Evaluate on test and log metrics + artifacts
            metrics = eval_and_log(best, X_test, y_test, run_name=f'train_{name}', model_name=name)

            # Log model
            mlflow.sklearn.log_model(best, artifact_path=f'model_{name}')
            # Save a local copy too
            joblib.dump(best, f'model_{name}.joblib')
            mlflow.log_artifact(f'model_{name}.joblib')

    print('Entrenamiento completado.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--experiment', type=str, default='titanic_classification')
    parser.add_argument('--k', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    args = parser.parse_args()
    main(args)
