# Changelog

## [1.0.0] - 2026-10-17

### Añadido
- Dataset original de Titanic versionado en `datos/datos_ini/titanic.csv`.
- Dataset limpio en `datos/datos_limp/titanic_limpio.csv`.
- Script de limpieza en `fuentes/datos_prep.py`.
- Script de entrenamiento y registro en MLflow: `fuentes/train.py`.
- Notebook `fuentes/entrena.ipynb` con el flujo de trabajo completo.
- Documentación del proyecto en `README.md`.
- Requisitos del entorno en `requirements.txt`.

### Desarrollo
- Preparación y limpieza de variables.
- Selección de modelos: Regresión Logística y Random Forest.
- Validación cruzada para comparación objetiva.
- Ajuste de hiperparámetros usando GridSearchCV.
- Registro de parámetros, métricas y artefactos en MLflow.

### Resultado esperado
El proyecto permite comparar dos algoritmos y confirmar la mejor configuración según las métricas definidas. Los experimentos quedan registrados en MLflow para trazabilidad y reproducibilidad.

