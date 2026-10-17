"""
Script de entrenamiento y registro con MLflow.
Permite seleccionar modelo entre logistic_regression, random_forest y xgboost.
Registra parámetros, métricas y artefactos (modelo, reportes).
"""
import os
import argparse
import mlflow
import mlflow.sklearn
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fuentes.datos_prep import load_raw, basic_clean, save_clean


MODEL_MAP = {
    'logistic_regression': LogisticRegression(max_iter=1000),
    'random_forest': RandomForestClassifier(random_state=42),
    'xgboost': xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
}

PARAM_GRID = {
    'logistic_regression': {
        'clf__C': [0.01, 0.1, 1.0, 10.0]
    },
    'random_forest': {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [None, 5, 10]
    },
    'xgboost': {
        'clf__n_estimators': [50, 100],
        'clf__max_depth': [3, 5]
    }
}


def prepare_features(df: pd.DataFrame):
    # target
    y = df['Survived']
    X = df.drop(columns=['Survived'])
    return X, y


def evaluate_model(model, X, y):
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = 'accuracy'
    scores = cross_val_score(model, X, y, cv=skf, scoring=scoring)
    return scores


def main(args):
    # Ensure data is prepared
    if not os.path.exists('datos/datos_limp/titanic_clean.csv'):
        print('Clean data not found, running data prep...')
        df_raw = load_raw()
        df_clean = basic_clean(df_raw)
        save_clean(df_clean)
    else:
        df_clean = pd.read_csv('datos/datos_limp/titanic_clean.csv')

    X, y = prepare_features(df_clean)

    # Build pipeline
    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', MODEL_MAP[args.model])
    ])

    param_grid = PARAM_GRID[args.model]

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    grid = GridSearchCV(pipe, param_grid=param_grid, cv=skf, scoring='accuracy', n_jobs=1)

    # MLflow experiment
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=f"{args.model}_run") as run:
        mlflow.log_param('model', args.model)
        mlflow.log_param('param_grid', str(param_grid))

        grid.fit(X, y)
        best = grid.best_estimator_
        best_params = grid.best_params_
        best_score = grid.best_score_

        # Evaluate on cross-val (already best_score)
        mlflow.log_metric('cv_accuracy', float(best_score))

        # Additional metrics on predictions using cross_val
        # We'll compute CV predictions to derive precision/recall/f1
        from sklearn.model_selection import cross_val_predict
        y_pred = cross_val_predict(best, X, y, cv=skf)
        acc = accuracy_score(y, y_pred)
        prec = precision_score(y, y_pred)
        rec = recall_score(y, y_pred)
        f1 = f1_score(y, y_pred)
        try:
            probas = cross_val_predict(best, X, y, cv=skf, method='predict_proba')[:,1]
            auc = roc_auc_score(y, probas)
        except Exception:
            auc = None

        mlflow.log_metric('accuracy', float(acc))
        mlflow.log_metric('precision', float(prec))
        mlflow.log_metric('recall', float(rec))
        mlflow.log_metric('f1', float(f1))
        if auc is not None:
            mlflow.log_metric('roc_auc', float(auc))

        # Log best params
        for k, v in best_params.items():
            mlflow.log_param(k, v)

        # Save model artifact
        os.makedirs('reports', exist_ok=True)
        model_path = f'reports/{args.model}_best_model.joblib'
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path, artifact_path='models')

        # Save cleaned dataset used
        mlflow.log_artifact('datos/datos_limp/titanic_clean.csv', artifact_path='data')

        print('Run finished. Best CV accuracy:', best_score)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='random_forest', choices=list(MODEL_MAP.keys()))
    parser.add_argument('--experiment-name', dest='experiment_name', type=str, default='Titanic_Classification')
    args = parser.parse_args()
    main(args)
