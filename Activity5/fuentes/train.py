"""
train.py
Script de entrenamiento y registro en MLflow.
Ejemplo:
python Activity5/fuentes/train.py --model rf --n_iter 10 --mlflow_uri file:./mlruns
"""
import argparse
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score,roc_auc_score,precision_score,recall_score,f1_score
import os
import joblib
from datos_prep import prepare_for_model


def get_data(clean_path='Activity5/datos/datos_limp/titanic_clean.csv', raw_path='Activity5/datos/datos_ini/titanic.csv'):
    if not os.path.exists(clean_path):
        os.makedirs(os.path.dirname(clean_path), exist_ok=True)
        prepare_for_model(raw_path, clean_path)
    df = pd.read_csv(clean_path)
    X = df.drop(columns=['Survived'])
    y = df['Survived']
    return X, y


def make_pipeline(model_name):
    if model_name == 'lr':
        estimator = LogisticRegression(max_iter=1000)
        param_grid = {
            'clf__C': [0.01, 0.1, 1.0, 10.0],
            'clf__penalty': ['l2']
        }
    elif model_name == 'rf':
        estimator = RandomForestClassifier(random_state=42)
        param_grid = {
            'clf__n_estimators': [50, 100, 200],
            'clf__max_depth': [None, 5, 10]
        }
    else:
        raise ValueError('model must be lr or rf')

    pipe = Pipeline([
        ('scaler', StandardScaler()),
        ('clf', estimator)
    ])
    return pipe, param_grid


def evaluate(y_true, y_pred_proba, y_pred):
    acc = accuracy_score(y_true, y_pred)
    auc = roc_auc_score(y_true, y_pred_proba)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    return dict(accuracy=acc, roc_auc=auc, precision=prec, recall=rec, f1=f1)


def main(args):
    mlflow.set_tracking_uri(args.mlflow_uri)
    X, y = get_data()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=args.test_size, random_state=42, stratify=y)

    pipe, param_grid = make_pipeline(args.model)
    cv = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    gs = GridSearchCV(pipe, param_grid, cv=cv, scoring='roc_auc', n_jobs=-1, verbose=1)

    with mlflow.start_run():
        mlflow.log_param('model', args.model)
        mlflow.log_param('kfold', args.kfold)
        # fit
        gs.fit(X_train, y_train)
        best = gs.best_estimator_
        mlflow.log_params(gs.best_params_)

        # predictions
        y_pred = best.predict(X_test)
        y_proba = best.predict_proba(X_test)[:,1]
        metrics = evaluate(y_test, y_proba, y_pred)
        for k,v in metrics.items():
            mlflow.log_metric(k, v)

        # save model artifact
        model_path = f"model_{args.model}.pkl"
        joblib.dump(best, model_path)
        mlflow.log_artifact(model_path)
        mlflow.sklearn.log_model(best, f"sklearn-{args.model}")

        print('Logged run to MLflow with metrics:', metrics)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lr', choices=['lr','rf'])
    parser.add_argument('--kfold', type=int, default=5)
    parser.add_argument('--test_size', type=float, default=0.2)
    parser.add_argument('--mlflow_uri', type=str, default='file:./mlruns')
    args = parser.parse_args()
    main(args)
