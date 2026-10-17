"""train.py

Script ejecutable para entrenar modelos sobre Titanic, hacer búsqueda de hiperparámetros
y loggear resultados (parámetros, métricas y artefactos) en MLflow.

Uso (ejemplo):
python fuentes/train.py --model rf --n-splits 5 --experiment-name Titanic_Actividad5
"""

import os
import argparse
import joblib
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold, cross_validate, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

from fuentes.datos_prep import load_data, preprocess, download_dataset


def get_models_and_params():
    models = {
        'lr': (LogisticRegression(solver='liblinear', random_state=42), {
            'clf__C': [0.01, 0.1, 1.0, 10.0]
        }),
        'rf': (RandomForestClassifier(random_state=42), {
            'clf__n_estimators': [50, 100],
            'clf__max_depth': [None, 5, 10]
        })
    }
    return models


def run_training(model_key='rf', n_splits=5, experiment_name='Titanic_Actividad5', output_dir='outputs'):
    os.makedirs(output_dir, exist_ok=True)
    # Prepare data
    path = download_dataset()
    df = load_data(path)
    X, y = preprocess(df, training=True)

    # Preprocessing pipeline: scale numeric features
    numeric_features = ['Pclass','Age','SibSp','Parch','Fare']
    numeric_transformer = StandardScaler()

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features)
        ], remainder='passthrough'
    )

    models = get_models_and_params()
    if model_key not in models:
        raise ValueError(f"Modelo desconocido {model_key}. Opciones: {list(models.keys())}")

    clf, param_grid = models[model_key]

    pipeline = Pipeline(steps=[('pre', preprocessor), ('clf', clf)])

    # MLflow experiment
    mlflow.set_tracking_uri('file://' + os.path.abspath('mlruns'))
    mlflow.set_experiment(experiment_name)

    # Search with cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    gs = GridSearchCV(pipeline, param_grid=param_grid, cv=cv, scoring='accuracy', n_jobs=-1)

    with mlflow.start_run(run_name=f"train_{model_key}") as run:
        # Fit
        gs.fit(X, y)

        best = gs.best_estimator_
        best_params = gs.best_params_
        cv_results = gs.cv_results_

        # Predict on training via cross_val_predict or evaluate best on CV splits
        scores = cross_validate(best, X, y, cv=cv, scoring=['accuracy','precision','recall','f1'], n_jobs=-1)

        metrics = {
            'accuracy_mean': float(np.mean(scores['test_accuracy'])),
            'accuracy_std': float(np.std(scores['test_accuracy'])),
            'precision_mean': float(np.mean(scores['test_precision'])),
            'recall_mean': float(np.mean(scores['test_recall'])),
            'f1_mean': float(np.mean(scores['test_f1'])),
        }

        # Log params and metrics
        mlflow.log_params(best_params)
        mlflow.log_metrics(metrics)

        # Save best model artifact
        model_path = os.path.join(output_dir, f"best_model_{model_key}.pkl")
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path, artifact_path='models')

        # Save a small json with summary
        summary = {
            'model': model_key,
            'best_params': best_params,
            'metrics': metrics
        }
        summary_path = os.path.join(output_dir, 'summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path)

        print("Entrenamiento completado. Run id:", run.info.run_id)
        print("Metrics:", metrics)
    return run.info.run_id


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='rf', help='lr or rf')
    parser.add_argument('--n-splits', type=int, default=5)
    parser.add_argument('--experiment-name', type=str, default='Titanic_Actividad5')
    args = parser.parse_args()
    run_training(model_key=args.model, n_splits=args.n_splits, experiment_name=args.experiment_name)
