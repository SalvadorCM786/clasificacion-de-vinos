import numpy as np
import streamlit as st
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix

st.write(''' # Diplomado Superior en Ciencia y Analítica de Datos ''')
st.write(''' Módulo IV: Big Data ''')
st.write(''' ***Profesora:*** DRA. EN I. Ana Estela Pérez Mejía
         
***Alumno:*** Salvador Calderón Martínez ''')

#Pactica
st.write(''' # Práctica 2: Streamlit app ''')
st.write(''' Instrucciones: Crear una app en Streamlit de algún modelo de predicción, puede ser clasificación o regresión.''')

# Descripción del modelo
st.write(''' # Modelo de Clasificación de Vinos ''')
st.write('''Este modelo predice el origen de un vino a partir de su composición
química, usando un Random Forest entrenado con un pipeline de escalado y validación cruzada.
El dataset clasifica los vinos en 3 clases distintas.''')

# dataset
wine = load_wine()
X = pd.DataFrame(wine.data, columns=wine.feature_names)
Y = pd.Series(wine.target)
nombres_clases = ['Vino tipo 1', 'Vino tipo 2', 'Vino tipo 3']

# Nombre de las columnas
nombres_columnas_es = {
    'alcohol': 'Alcohol',
    'malic_acid': 'Ácido málico',
    'ash': 'Ceniza',
    'alcalinity_of_ash': 'Alcalinidad de ceniza',
    'magnesium': 'Magnesio',
    'total_phenols': 'Fenoles totales',
    'flavanoids': 'Flavonoides',
    'nonflavanoid_phenols': 'Fenoles no flavonoides',
    'proanthocyanins': 'Proantocianinas',
    'color_intensity': 'Intensidad de color',
    'hue': 'Tono',
    'od280/od315_of_diluted_wines': 'OD280/OD315',
    'proline': 'Prolina'
}
X = X.rename(columns=nombres_columnas_es)

st.header('Datos de evaluación')
st.caption('''Ingresa los valores de composición química del vino que quieres clasificar. '''
           '''Estos datos serán procesados por el modelo para predecir a cuál de los 3 cultivares pertenece.''')

def user_input_features():
    col1, col2 = st.columns(2)

    with col1:
        alcohol = st.number_input('Alcohol:', min_value=0.0, max_value=20.0, value=13.0, step=0.1)
        malic_acid = st.number_input('Ácido málico:', min_value=0.0, max_value=10.0, value=2.3, step=0.1)
        ash = st.number_input('Ceniza:', min_value=0.0, max_value=5.0, value=2.4, step=0.1)
        alcalinity = st.number_input('Alcalinidad de ceniza:', min_value=0.0, max_value=40.0, value=19.0, step=0.1)
        magnesium = st.number_input('Magnesio:', min_value=0.0, max_value=200.0, value=100.0, step=1.0)
        phenols = st.number_input('Fenoles totales:', min_value=0.0, max_value=5.0, value=2.3, step=0.1)
        flavanoids = st.number_input('Flavonoides:', min_value=0.0, max_value=6.0, value=2.0, step=0.1)

    with col2:
        nonflavanoid = st.number_input('Fenoles no flavonoides:', min_value=0.0, max_value=1.0, value=0.3, step=0.01)
        proanthocyanins = st.number_input('Proantocianinas:', min_value=0.0, max_value=4.0, value=1.6, step=0.1)
        color_intensity = st.number_input('Intensidad de color:', min_value=0.0, max_value=15.0, value=5.0, step=0.1)
        hue = st.number_input('Tono:', min_value=0.0, max_value=2.0, value=1.0, step=0.01)
        od280 = st.number_input('OD280/OD315:', min_value=0.0, max_value=5.0, value=2.6, step=0.1)
        proline = st.number_input('Prolina:', min_value=0.0, max_value=2000.0, value=750.0, step=10.0)

    user_input_data = {
        'Alcohol': alcohol, 'Ácido málico': malic_acid, 'Ceniza': ash,
        'Alcalinidad de ceniza': alcalinity, 'Magnesio': magnesium,
        'Fenoles totales': phenols, 'Flavonoides': flavanoids,
        'Fenoles no flavonoides': nonflavanoid, 'Proantocianinas': proanthocyanins,
        'Intensidad de color': color_intensity, 'Tono': hue,
        'OD280/OD315': od280, 'Prolina': proline
    }

    features = pd.DataFrame(user_input_data, index=[0])
    return features

df = user_input_features()

# Separar en entrenamiento y prueba
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=0, stratify=Y)

# Pipeline: escalado + Random Forest
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, max_depth=6, random_state=0, n_jobs=-1))
])

pipeline.fit(X_train, Y_train)

# Métricas sobre datos de prueba
Y_pred_test = pipeline.predict(X_test)
accuracy = accuracy_score(Y_test, Y_pred_test)

# Validación cruzada (5 folds) sobre datos de entrenamiento
cv_scores = cross_val_score(pipeline, X_train, Y_train, cv=5, scoring='accuracy')

# Predicción sobre la entrada del usuario
prediction = pipeline.predict(df)[0]
prediction_proba = pipeline.predict_proba(df)[0]

st.subheader('Predicción')
st.caption('Muestra el cultivar de vino que el modelo considero más probable de acuerdo con los valores ingresados.')
st.success(f'Cultivar predicho: **{nombres_clases[prediction]}**')

st.subheader('Probabilidad por clase')
st.caption('Indica qué tan seguro está el modelo de cada posible clase. Una probabilidad en una sola '
           'clase indica mayor confianza en la predicción, pero probabilidades repartidas entre varias clases '
           'indica que no esta bien definido (Se adjunta en la entrega de la practica un archivo .txt con 3 ejemplos por clase).')
proba_df = pd.DataFrame({'Cultivar': nombres_clases, 'Probabilidad': prediction_proba})
st.bar_chart(proba_df.set_index('Cultivar'))

st.subheader('Desempeño del modelo')
st.caption('Resume qué tan bien clasifica el modelo en general, usando datos que no vio durante el '
           'entrenamiento. La precisión de prueba mide los aciertos en una sola partición de datos, mientras '
           'que la validación cruzada (5-fold) promedia el desempeño en 5 particiones distintas, dando una '
           'medida más confiable.')
col1, col2 = st.columns(2)
col1.metric('Precisión (prueba)', f'{accuracy:.2%}')
col2.metric('Precisión promedio (5-fold CV)', f'{cv_scores.mean():.2%}')

st.subheader('Matriz de confusión (datos de prueba)')
st.caption('Compara las predicciones del modelo contra las clases reales en el conjunto de prueba. La diagonal '
           'muestra los aciertos, los valores fuera de la diagonal indican en qué clases se equivocó el modelo '
           'y con cuál las confundió.')
cm = confusion_matrix(Y_test, Y_pred_test)
cm_df = pd.DataFrame(cm, index=[f'Real: {c}' for c in nombres_clases],
                      columns=[f'Pred: {c}' for c in nombres_clases])
st.dataframe(cm_df)

st.subheader('Importancia de variables')
st.caption('Muestra qué tanto contribuye cada característica química a la decisión del modelo. Las variables '
           'con barras más altas son las que más influyen al distinguir entre los tres cultivares de vino.')
importancias = pd.DataFrame({
    'Variable': X.columns,
    'Importancia': pipeline.named_steps['classifier'].feature_importances_
}).sort_values('Importancia', ascending=False)
st.bar_chart(importancias.set_index('Variable'))