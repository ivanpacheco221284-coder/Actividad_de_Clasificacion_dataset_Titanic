import json
from pathlib import Path

import mlflow
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split

from fuentes.datos_prep import CLEAN_DATA_PATH, build_preprocessor, clean_dataset, load_dataset, prepare_features_for_training, save_clean_dataset

ROOT_DIR = Path(__file__).resolve().parent.parent
TRACKING_URI = ROOT_DIR / "mlruns"
RESULTS_DIR = ROOT_DIR / "resultados"
RESULTS_DIR.mkdir(exist_ok=True)


def compute_metrics(y_true: pd.Series, y_pred: np.ndarray, y_proba: np.ndarray | None = None) -> dict:
    """Calcula las métricas estándar para clasificación binaria."""
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }
    if y_proba is not None:
        metrics["roc_auc"] = roc_auc_score(y_true, y_proba)
    return metrics


def plot_confusion_matrix(y_true: pd.Series, y_pred: np.ndarray, model_name: str) -> str:
    """Genera y devuelve la ruta de la matriz de confusión."""
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_title(f"Matriz de confusión - {model_name}")
    ax.set_xlabel("Predicción")
    ax.set_ylabel("Real")
    classes = [0, 1]
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center", color="white" if cm[i, j] > cm.max() / 2 else "black")

    fig.tight_layout()
    path = RESULTS_DIR / f"{model_name}_confusion_matrix.png"
    fig.savefig(path, dpi=200)
    plt.close(fig)
    return str(path)


def create_model_specs():
    """Define los modelos a evaluar con sus grids de hiperparámetros."""
    return {
        "logistic_regression": {
            "estimator": LogisticRegression(max_iter=2000, random_state=42),
            "param_grid": {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__solver": ["liblinear", "lbfgs"],
                "classifier__class_weight": [None, "balanced"],
            },
        },
        "random_forest": {
            "estimator": RandomForestClassifier(random_state=42),
            "param_grid": {
                "classifier__n_estimators": [100, 200],
                "classifier__max_depth": [3, 5, None],
                "classifier__min_samples_leaf": [1, 2, 5],
            },
        },
    }


def train_and_log_models():
    """Ejecuta validación cruzada, evaluación y registro de experimentos en MLflow."""
    df = load_dataset()
    clean_df = clean_dataset(df)
    save_clean_dataset(clean_df)

    X, y = prepare_features_for_training(clean_df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri(f"file://{TRACKING_URI}")
    mlflow.set_experiment("Titanic_Classification")

    model_specs = create_model_specs()
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for model_name, config in model_specs.items():
        pipeline = __import__("sklearn.pipeline", fromlist=["Pipeline"]).Pipeline(
            steps=[
                ("preprocessor", build_preprocessor()),
                ("classifier", config["estimator"]),
            ]
        )

        search = GridSearchCV(
            estimator=pipeline,
            param_grid=config["param_grid"],
            scoring="roc_auc",
            cv=cv,
            n_jobs=-1,
            verbose=0,
            refit=True,
        )

        search.fit(X_train, y_train)
        y_pred = search.predict(X_test)
        y_proba = search.predict_proba(X_test)[:, 1]
        metrics = compute_metrics(y_test, y_pred, y_proba)

        with mlflow.start_run(run_name=f"{model_name}_best") as run:
            mlflow.log_param("model_name", model_name)
            mlflow.log_params(search.best_params_)
            mlflow.log_metric("cv_roc_auc", search.best_score_)
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)

            figure_path = plot_confusion_matrix(y_test, y_pred, model_name)
            mlflow.log_artifact(figure_path, artifact_path="plots")
            mlflow.log_artifact(str(CLEAN_DATA_PATH), artifact_path="dataset")
            mlflow.sklearn.log_model(search.best_estimator_, artifact_path="model")

            run_data = {
                "model": model_name,
                "cv_best_score": round(search.best_score_, 6),
                "metrics": metrics,
                "best_params": search.best_params_,
            }
            with open(RESULTS_DIR / f"{model_name}_resultados.json", "w", encoding="utf-8") as f:
                json.dump(run_data, f, ensure_ascii=False, indent=2)
            mlflow.log_artifact(str(RESULTS_DIR / f"{model_name}_resultados.json"), artifact_path="results")

        print(f"Modelo {model_name}: {metrics}")

    print(f"MLflow tracking en: {TRACKING_URI}")


if __name__ == "__main__":
    train_and_log_models()
