# Clasificación de Vinos con Random Forest

Aplicación web interactiva construida con **Streamlit** que predice el cultivar (origen) de un vino a partir de su composición química, utilizando un modelo de **Random Forest**.

##  Descripción

Este proyecto entrena un modelo de clasificación multiclase sobre el dataset clásico **Wine** (scikit-learn), el cual agrupa vinos en 3 cultivares distintos según 13 características químicas medidas en laboratorio (alcohol, acidez, fenoles, intensidad de color, entre otras).

La aplicación permite ingresar los valores de un vino nuevo y obtener en tiempo real:
- La clase predicha (Vino tipo 1, 2 o 3)
- La probabilidad asociada a cada clase
- Métricas de desempeño del modelo (precisión y validación cruzada)
- La matriz de confusión sobre los datos de prueba
- Un gráfico con la importancia de cada variable en la predicción

## Características

- **Modelo:** Random Forest Classifier (100 árboles, profundidad máxima 6)
- **Pipeline:** Escalado de datos (`StandardScaler`) + clasificador, evitando fuga de datos
- **Validación:** División train/test estratificada (80/20) + validación cruzada de 5 folds
- **Interfaz:** Formulario interactivo en español con 13 variables de entrada
- **Visualizaciones:** Gráfico de probabilidades, matriz de confusión e importancia de variables

## Herramientas

- Python 3.14.5
- [Streamlit](https://streamlit.io/) — interfaz web
- [scikit-learn](https://scikit-learn.org/) — modelo de machine learning
- [pandas](https://pandas.pydata.org/) / [numpy](https://numpy.org/) — manejo de datos

