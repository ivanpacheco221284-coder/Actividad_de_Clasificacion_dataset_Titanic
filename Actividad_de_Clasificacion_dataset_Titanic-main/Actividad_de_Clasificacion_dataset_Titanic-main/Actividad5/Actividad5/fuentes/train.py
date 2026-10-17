import json
import os
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from datos_prep import clean_dataset, load_raw_data


BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DATA_PATH = BASE_DIR / "datos" / "datos_ini" / "titanic.csv"
CLEAN_DATA_PATH = BASE_DIR / "datos" / "datos_limp" / "titanic_limpio.csv"
MLFLOW_URI = f"file://{BASE_DIR / 'mlruns'}"


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop",
    )

    return preprocessor


def evaluate_model(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
    }

    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_test, y_proba)

    return metrics, classification_report(y_test, y_pred, output_dict=True)


def train_and_log(model_name, model, X_train, X_test, y_train, y_test, param_grid=None):
    if param_grid:
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
        grid = GridSearchCV(model, param_grid=param_grid, cv=cv, scoring="f1", n_jobs=-1)
        grid.fit(X_train, y_train)
        best_model = grid.best_estimator_
        best_params = grid.best_params_
    else:
        best_model = model
        best_model.fit(X_train, y_train)
        best_params = model.get_params()

    metrics, report = evaluate_model(best_model, X_test, y_test)

    with mlflow.start_run(run_name=model_name):
        mlflow.log_params({k: str(v) for k, v in best_params.items() if not isinstance(v, (list, dict, tuple))})
        for key, value in metrics.items():
            mlflow.log_metric(key, float(value))

        mlflow.log_dict(report, f"{model_name}_classification_report.json")

        # artefacto extra para evidencia
        artifact_path = BASE_DIR / "mlruns" / f"{model_name}_report.txt"
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        artifact_path.write_text(json.dumps({"metrics": metrics, "classification_report": report}, indent=2, ensure_ascii=False))
        mlflow.log_artifact(str(artifact_path), artifact_path="reports")

        mlflow.sklearn.log_model(best_model, artifact_path="model")

    return best_model, metrics


def main():
    # configuración
    os.makedirs(BASE_DIR / "mlruns", exist_ok=True)
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment("Titanic_Classification")

    if not RAW_DATA_PATH.exists():
        raise FileNotFoundError(f"Dataset no encontrado: {RAW_DATA_PATH}")

    raw_df = load_raw_data(RAW_DATA_PATH)
    df_clean = clean_dataset(raw_df)
    df_clean.to_csv(CLEAN_DATA_PATH, index=False)

    target = "Survived"
    y = df_clean[target]
    X = df_clean.drop(columns=[target])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor(X_train)

    log_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LogisticRegression(max_iter=1000, random_state=42)),
        ]
    )

    rf_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", RandomForestClassifier(random_state=42, n_estimators=200)),
        ]
    )

    log_param_grid = {
        "model__C": [0.1, 1, 10],
        "model__solver": ["liblinear", "lbfgs"],
    }

    rf_param_grid = {
        "model__n_estimators": [100, 200],
        "model__max_depth": [None, 5, 10],
        "model__min_samples_split": [2, 5],
    }

    train_and_log("logistic_regression", log_model, X_train, X_test, y_train, y_test, param_grid=log_param_grid)
    train_and_log("random_forest", rf_model, X_train, X_test, y_train, y_test, param_grid=rf_param_grid)

    print(f"Se ha limpiado y guardado el dataset en: {CLEAN_DATA_PATH}")
    print(f"MLflow tracking URI: {MLFLOW_URI}")


if __name__ == "__main__":
    main()
