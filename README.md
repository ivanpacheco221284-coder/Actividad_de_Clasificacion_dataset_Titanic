# Actividad de Clasificación: Titanic

Repositorio: Actividad_de_Clasificacion_dataset_Titanic

Este repositorio contiene el código, notebook y guías para reproducir la actividad solicitada: preparación de datos, entrenamiento y ajuste de modelos, y registro de experimentos con MLflow.

Estructura del repositorio

Actividad_de_Clasificacion_dataset_Titanic/
├── datos/
│   ├── datos_ini/ (se descarga titanic.csv al ejecutar los scripts)
│   └── datos_limp/ (placeholder)
├── fuentes/
│   ├── entrena.ipynb (notebook listo para Colab)
│   ├── datos_prep.py (funciones de limpieza y carga)
│   └── train.py (script para entrenamiento y registro en MLflow)
├── README.md (este archivo)
├── CHANGELOG.md
└── requirements.txt

Requisitos

Instalar (por ejemplo en Colab ejecuta):

!pip install scikit-learn pandas mlflow xgboost joblib seaborn matplotlib

Instrucciones de reproducción (Colab o local)

1. Abrir `fuentes/entrena.ipynb` en Colab y ejecutar las celdas en orden. Esto:
   - instalará dependencias
   - descargará el dataset Titanic y lo almacenará en datos/datos_ini/titanic.csv
   - ejecutará `fuentes/train.py` para RandomForest y LogisticRegression
   - registrará resultados en `mlruns/` (directorio local)

2. Alternativamente ejecutar localmente:
   - python3 -m venv venv && source venv/bin/activate
   - pip install -r requirements.txt
   - python3 fuentes/train.py --model rf --n-splits 5 --experiment-name Titanic_Actividad5
   - python3 fuentes/train.py --model lr --n-splits 5 --experiment-name Titanic_Actividad5

MLflow

El tracking URI por defecto es el directorio `mlruns/` dentro del repo (archivo). Para arrancar la interfaz localmente:

mlflow ui --backend-store-uri file://$(pwd)/mlruns

Evaluación y métricas

Se registran métricas promedios por validación cruzada: accuracy, precision, recall y f1. Además se guardan los artefactos `best_model_{model}.pkl` y `summary.json` con parámetros y métricas.

Decisiones y consideraciones

- Se usó LogisticRegression y RandomForest para comparación (clasificación binaria: Survived).
- Se hicieron transformaciones simples: imputación por mediana para Age/Fare y one-hot para Embarked.
- Se realizó búsqueda de hiperparámetros con GridSearchCV y validación estratificada.

Contacto

Repositorio creado para la entrega de la actividad. Si necesitas que suba resultados de MLflow (capturas de pantalla) o que ejecute entrenamientos remotos, indícalo.
