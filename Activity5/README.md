# Actividad 5 - Clasificación Titanic

Repositorio con la actividad de clasificación usando el dataset Titanic.

Estructura:

Actividad5/
│── datos/
│   ├── datos_ini/ (scripts para obtención del dataset)
│   └── datos_limp/ (salida de limpieza)
│── fuentes/
│   ├── entrena.ipynb
│   ├── datos_prep.py
│   └── train.py
│── README.md
│── CHANGELOG.md

Reproducción rápida:

1. Clona el repositorio.
2. Crear entorno y instalar dependencias: pip install -r Activity5/requirements.txt
3. Descargar dataset y limpiar:
   - Ejecuta: python Activity5/datos/datos_ini/get_titanic.py
   - Ejecuta: python Activity5/fuentes/datos_prep.py Activity5/datos/datos_ini/titanic.csv Activity5/datos/datos_limp/titanic_clean.csv
4. Entrenar y registrar con MLflow (local):
   - Ejecuta: python Activity5/fuentes/train.py --model rf --mlflow_uri file:./Activity5/mlruns
   - O desde notebook: abrir Activity5/fuentes/entrena.ipynb (Colab) y ejecutar celdas.

MLflow:
Se configura para usar almacenamiento local por defecto (Activity5/mlruns). Para usar un servidor remoto, cambie --mlflow_uri a la dirección correspondiente.
