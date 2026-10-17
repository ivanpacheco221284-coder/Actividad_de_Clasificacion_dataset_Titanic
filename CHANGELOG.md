# Changelog

## v1.0.0 - 2026-10-17

### Añadido
- Estructura del repositorio para la actividad de clasificación.
- Dataset Titanic en `datos/datos_ini/titanic.csv`.
- Script de limpieza y preparación `fuentes/datos_prep.py`.
- Script de entrenamiento y registro en MLflow `fuentes/train.py`.
- Notebook `fuentes/entrena.ipynb` con flujo de trabajo completo.
- Documentación de ejecución y reproducción en `README.md`.
- Archivo de dependencias `requirements.txt`.
- Configuración de Git ignore para artefactos de Python y MLflow.

### Metodología aplicada
- Preparación y versionado del dataset.
- Limpieza de variables con imputación y codificación.
- Entrenamiento de dos modelos:
  - Regresión logística
  - Random Forest
- Ajuste de hiperparámetros con `GridSearchCV`.
- Validación cruzada estratificada (`StratifiedKFold` con 5 folds).
- Evaluación con métricas estándar.
- Registro de parámetros, métricas y artefactos en MLflow.

### Resultado esperado
- Reproducibilidad del entrenamiento.
- Comparación objetiva de modelos.
- Trazabilidad y almacenamiento de artefactos en MLflow.
