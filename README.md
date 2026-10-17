# Actividad de Clasificación - Titanic

Este repositorio contiene la actividad 5: preparación de datos, entrenamiento de modelos y registro con MLflow sobre el dataset Titanic.

Estructura:

Actividad5/
│── datos/
│   ├── datos_ini/ (dataset original)
│   └── datos_limp/ (datos limpios preparados)
│── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
│── README.md
│── CHANGELOG.md

Reproducción rápida:

1. Crear entorno virtual e instalar dependencias:

python -m venv .venv
source .venv/bin/activate  # o .venv\Scripts\activate en Windows
pip install -r requirements.txt

2. Descargar dataset:

python datos/download_data.py

3. Preparar datos y entrenar (ejemplo):

python fuentes/datos_prep.py
python fuentes/train.py --model random_forest --experiment-name Titanic_Classification

4. Ejecutar el notebook `fuentes/entrena.ipynb` en Colab o local.

MLflow:

- Por defecto, los runs se registran en mlruns/ localmente. Para verlos:

mlflow ui --port 5000

- En Colab o entorno remoto, configure MLflow tracking URI a un servidor remoto si desea.

Decisiones de limpieza:

- Se eliminaron columnas irrelevantes: PassengerId, Name, Ticket, Cabin.
- Se imputó Age y Fare con la mediana.
- Se codificó Sex a 0/1 y se aplicó one-hot a Embarked.
- Se añadió FamilySize.

Modelos probados:

- Regresión logística (scikit-learn)
- Random Forest (scikit-learn)
- XGBoost (opcional)

Métricas registradas: accuracy, precision, recall, f1, roc_auc (si está disponible).

