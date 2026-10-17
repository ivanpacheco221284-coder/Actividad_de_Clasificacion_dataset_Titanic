# Actividad 5 - Clasificación con Titanic

Descripción
Este repositorio contiene una actividad de clasificación sobre el dataset Titanic. El objetivo es preparar los datos, entrenar y comparar al menos dos algoritmos, ajustar hiperparámetros, evaluar con validación cruzada y registrar los experimentos con MLflow.

Objetivo
Preparar los datos y documentar las decisiones de limpieza.
entrenar al menos 2 modelos de clasificación: Regresión Logística y Random Forest.
Realizar ajuste de hiperparámetros con validación cruzada.
Registrar parámetros, métricas y artefactos en MLflow.
Generar una guía que permita reproducir el flujo completo.
Estructura del repositorio
Actividad5/ ├── datos/ │ ├── datos_ini/ │ │ └── titanic.csv │ └── datos_limp/ │ └── titanic_limpio.csv ├── fuentes/ │ ├── entrena.ipynb │ ├── datos_prep.py │ └── train.py ├── mlruns/ │ └── README.md ├── README.md ├── CHANGELOG.md ├── requirements.txt └── .gitignore

Dataset

Se utiliza un dataset público de Titanic con variables como Pclass, Sex, Age, SibSp, Fare, Embarked y la variable objetivo Survived. Se incluye una versión original y una versión limpia.

Requisitos

Python >= 3.10
pip install -r requirements.txt
Instalación
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
Ejecución del entrenamiento
Desde la raíz del proyecto:

python Actividad5/fuentes/train.py
O abrir el notebook:

jupyter notebook Actividad5/fuentes/entrena.ipynb
MLflow
La configuración local usa MLflow con un tracking URI local en la carpeta Actividad5/mlruns.

Para verificar resultados:

mlflow ui --backend-store-uri file:///$(pwd)/Actividad5/mlruns
Decisiones de limpieza
Se eliminan columnas no útiles para el modelo (Name, Ticket, Cabin y PassengerId).
Se imputan valores faltantes en variables numéricas y categóricas.
Se codifican variables categóricas mediante OneHotEncoder.
Se mantiene la variable objetivo Survived como binaria.
Se usa validación cruzada (cross_val_score o GridSearchCV) y se comparan los modelos.
Métricas empleadas
Accuracy
Precision
Recall
F1-score
ROC AUC
Criterios de calidad de la actividad
La solución cumple con:

preparación de datos y versionamiento,
entrenamiento de dos modelos,
ajuste de hiperparámetros,
evaluación cruzada,
registro de experimentos en MLflow,
documentación del proyecto.
Referencias
Dataset Titanic: https://github.com/datasciencedojo/datasets/blob/master/titanic.csv
MLflow: https://mlflow.org/
Scikit-learn: https://scikit-learn.org/
