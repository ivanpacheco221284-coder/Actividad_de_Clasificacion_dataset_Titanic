# Actividad de Clasificación - Titanic

Este repositorio contiene una solución completa para la actividad de clasificación del dataset Titanic. Incluye:
- preparación y versionado de datos
- limpieza y exploración
- entrenamiento de dos modelos de clasificación
- ajuste de hiperparámetros mediante validación cruzada
- registro del experimento en MLflow

## Estructura del repositorio

```text
Actividad5/
│── datos/
│   ├── datos_ini/
│   │   └── titanic.csv
│   └── datos_limp/
│       └── titanic_limpio.csv
│── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
│── README.md
│── CHANGELOG.md
│── requirements.txt
│── .gitignore
└── mlruns/
```

## Objetivo

Preparar los datos, entrenar modelos de clasificación, ajustar hiperparámetros y registrar resultados con MLflow para comparar el rendimiento de dos algoritmos sobre el dataset Titanic.

## Dataset

La fuente original es el dataset público Titanic disponible en el paquete `seaborn` y también se incluye una versión local en `datos/datos_ini/titanic.csv`.

## Requisitos

- Python 3.10+
- Jupyter Notebook
- Pip

## Instalación

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Ejecución del entrenamiento

```bash
python fuentes/train.py
```

Esto realizará:
- limpieza de datos
- división train/test
- validación cruzada con `GridSearchCV`
- evaluación con métricas estandarizadas
- registro de parámetros, métricas y artefactos en MLflow

## Ejecución del notebook

```bash
jupyter notebook fuentes/entrena.ipynb
```

## MLflow

La URI local de tracking es:

```bash
file:///.../Actividad_de_Clasificacion_dataset_Titanic/mlruns
```

Para visualizar la interfaz local:

```bash
mlflow ui --backend-store-uri file://$PWD/mlruns
```

Luego acceder a http://localhost:5000

## Modelos evaluados

1. Regresión logística
2. Random Forest

## Métricas

- accuracy
- precision
- recall
- f1_score
- roc_auc

## Decisiones de diseño

- Se eliminan columnas irrelevantes para la predicción (`PassengerId`, `Name`, `Ticket`, `Cabin`).
- Los valores faltantes se imputan con estrategias razonables por tipo de dato.
- Se aplica un preprocesamiento con `OneHotEncoder` para variables categóricas.
- Las variables numéricas se escalan cuando es necesario.
- Se usa validación cruzada estratificada con 5 folds.

## Contribución

La actividad está enfocada en reproducibilidad, trazabilidad y comparabilidad de resultados, cumpliendo con la lógica de su rubrica.
