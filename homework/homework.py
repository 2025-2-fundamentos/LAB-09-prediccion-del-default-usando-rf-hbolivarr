# flake8: noqa: E501
#
# En este dataset se desea pronosticar el default (pago) del cliente el próximo
# mes a partir de 23 variables explicativas.
#
#   LIMIT_BAL: Monto del credito otorgado. Incluye el credito individual y el
#              credito familiar (suplementario).
#         SEX: Genero (1=male; 2=female).
#   EDUCATION: Educacion (0=N/A; 1=graduate school; 2=university; 3=high school; 4=others).
#    MARRIAGE: Estado civil (0=N/A; 1=married; 2=single; 3=others).
#         AGE: Edad (years).
#       PAY_0: Historia de pagos pasados. Estado del pago en septiembre, 2005.
#       PAY_2: Historia de pagos pasados. Estado del pago en agosto, 2005.
#       PAY_3: Historia de pagos pasados. Estado del pago en julio, 2005.
#       PAY_4: Historia de pagos pasados. Estado del pago en junio, 2005.
#       PAY_5: Historia de pagos pasados. Estado del pago en mayo, 2005.
#       PAY_6: Historia de pagos pasados. Estado del pago en abril, 2005.
#   BILL_AMT1: Historia de pagos pasados. Monto a pagar en septiembre, 2005.
#   BILL_AMT2: Historia de pagos pasados. Monto a pagar en agosto, 2005.
#   BILL_AMT3: Historia de pagos pasados. Monto a pagar en julio, 2005.
#   BILL_AMT4: Historia de pagos pasados. Monto a pagar en junio, 2005.
#   BILL_AMT5: Historia de pagos pasados. Monto a pagar en mayo, 2005.
#   BILL_AMT6: Historia de pagos pasados. Monto a pagar en abril, 2005.
#    PAY_AMT1: Historia de pagos pasados. Monto pagado en septiembre, 2005.
#    PAY_AMT2: Historia de pagos pasados. Monto pagado en agosto, 2005.
#    PAY_AMT3: Historia de pagos pasados. Monto pagado en julio, 2005.
#    PAY_AMT4: Historia de pagos pasados. Monto pagado en junio, 2005.
#    PAY_AMT5: Historia de pagos pasados. Monto pagado en mayo, 2005.
#    PAY_AMT6: Historia de pagos pasados. Monto pagado en abril, 2005.
#
# La variable "default payment next month" corresponde a la variable objetivo.
#
# El dataset ya se encuentra dividido en conjuntos de entrenamiento y prueba
# en la carpeta "files/input/".
#
# Los pasos que debe seguir para la construcción de un modelo de
# clasificación están descritos a continuación.
#
#
# Paso 1.
# Realice la limpieza de los datasets:
# - Renombre la columna "default payment next month" a "default".
# - Remueva la columna "ID".
# - Elimine los registros con informacion no disponible.
# - Para la columna EDUCATION, valores > 4 indican niveles superiores
#   de educación, agrupe estos valores en la categoría "others".
# - Renombre la columna "default payment next month" a "default"
# - Remueva la columna "ID".
#
#
# Paso 2.
# Divida los datasets en x_train, y_train, x_test, y_test.
#
#
# Paso 3.
# Cree un pipeline para el modelo de clasificación. Este pipeline debe
# contener las siguientes capas:
# - Transforma las variables categoricas usando el método
#   one-hot-encoding.
# - Ajusta un modelo de bosques aleatorios (rando forest).
#
#
# Paso 4.
# Optimice los hiperparametros del pipeline usando validación cruzada.
# Use 10 splits para la validación cruzada. Use la función de precision
# balanceada para medir la precisión del modelo.
#
#
# Paso 5.
# Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
# Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.
#
#
# Paso 6.
# Calcule las metricas de precision, precision balanceada, recall,
# y f1-score para los conjuntos de entrenamiento y prueba.
# Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# Este diccionario tiene un campo para indicar si es el conjunto
# de entrenamiento o prueba. Por ejemplo:
#
# {'dataset': 'train', 'precision': 0.8, 'balanced_accuracy': 0.7, 'recall': 0.9, 'f1_score': 0.85}
# {'dataset': 'test', 'precision': 0.7, 'balanced_accuracy': 0.6, 'recall': 0.8, 'f1_score': 0.75}
#
#
# Paso 7.
# Calcule las matrices de confusion para los conjuntos de entrenamiento y
# prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
# del archivo es un diccionario con las metricas de un modelo.
# de entrenamiento o prueba. Por ejemplo:
#
# {'type': 'cm_matrix', 'dataset': 'train', 'true_0': {"predicted_0": 15562, "predicte_1": 666}, 'true_1': {"predicted_0": 3333, "predicted_1": 1444}}
# {'type': 'cm_matrix', 'dataset': 'test', 'true_0': {"predicted_0": 15562, "predicte_1": 650}, 'true_1': {"predicted_0": 2490, "predicted_1": 1420}}
#

import zipfile
import pickle
import gzip
import json
import os
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    make_scorer,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# Función para limpiar y preprocesar los datos
def limpieza(df):
    df = df.rename(columns={'default payment next month':'default'})
    df = df.drop(columns='ID')
    df = df.dropna()
    df['EDUCATION'] = [4 if i >=4 else i for i in df['EDUCATION']]
    return df

# División de los datasets
def div_train_test(df):
    x = df.drop(columns='default')
    y = df['default']
    return x, y

# Construcción del pipeline
def pipeline(x_train, y_train):

    cat_columns = ['SEX', 'EDUCATION', 'MARRIAGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_4', 'PAY_5', 'PAY_6']
    num_columns = [col for col in x_train.columns if col not in cat_columns]
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), cat_columns),
            ('num', 'passthrough', num_columns)
        ]
    )
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])
    param_grid = {
        'classifier__n_estimators': [50, 100, 200],
        'classifier__max_depth': [None, 5, 10, 20],
        'classifier__min_samples_split': [2, 5, 10],
        'classifier__min_samples_leaf': [1, 2, 4]
    }
    scoring = make_scorer(balanced_accuracy_score)
    # skf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
    grid_search = GridSearchCV(
        pipe,
        param_grid=param_grid,
        cv=10,
        scoring=scoring,
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(x_train, y_train)

    return grid_search


# Guardar el modelo
def guardar_modelo(modelo):
    ruta = "files/models/model.pkl.gz"
    with gzip.open(ruta, 'wb') as f:
        pickle.dump(modelo, f)

# Calcular métricas y guardar
def calcular_metricas_y_guardar(modelo, x_train, y_train, x_test, y_test):

    y_pred_train = modelo.predict(x_train)
    y_pred_test = modelo.predict(x_test)

    def obtener_metricas(y_true, y_pred, dataset_nombre):
        return {
            'dataset': dataset_nombre,
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0)
        }

    resultados = [
        obtener_metricas(y_train, y_pred_train, 'train'),
        obtener_metricas(y_test, y_pred_test, 'test')
    ]

    ruta = 'files/output/metrics.json'
    with open(ruta, 'w') as f:
        for fila in resultados:
            f.write(json.dumps(fila) + '\n')


# Cálculo de matrices de confusión
def matriz(modelo, x_train, y_train, x_test, y_test, ruta='files/output/metrics.json'):
    y_pred_train = modelo.predict(x_train)
    y_pred_test = modelo.predict(x_test)
    def obtener_cm_dict(y_true, y_pred, dataset_nombre):
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        return {
            'type': 'cm_matrix',
            'dataset': dataset_nombre,
            'true_0': {
                'predicted_0': int(cm[0, 0]),
                'predicted_1': int(cm[0, 1])
            },
            'true_1': {
                'predicted_0': int(cm[1, 0]),
                'predicted_1': int(cm[1, 1])
            }
        }
    matrices = [
        obtener_cm_dict(y_train, y_pred_train, 'train'),
        obtener_cm_dict(y_test, y_pred_test, 'test')
    ]
    with open(ruta, 'a') as f:
        for fila in matrices:
            f.write(json.dumps(fila) + '\n')

    

# ------------------- EJECUCIÓN PRINCIPAL -------------------

if __name__ == "__main__":

    os.makedirs("files/models", exist_ok=True)
    os.makedirs("files/output", exist_ok=True)

    with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as comp:
        train_data = comp.namelist()[0]
        with comp.open(train_data) as arch:
            df_train = pd.read_csv(arch)

    with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as comp:
        test_data = comp.namelist()[0]
        with comp.open(test_data) as arch:
            df_test = pd.read_csv(arch)

    # limpieza 
    df_train = limpieza(df_train)
    df_test = limpieza(df_test)

    # división en x, y
    x_train, y_train = div_train_test(df_train)
    x_test, y_test = div_train_test(df_test)

    # entrenamiento y optimización
    grid = pipeline(x_train, y_train)
    modelo = grid.best_estimator_

    # guardar modelo
    guardar_modelo(grid)

    # calcular métricas y guardar
    calcular_metricas_y_guardar(modelo, x_train, y_train, x_test, y_test)

    # matrices de confusión
    matriz(modelo, x_train, y_train, x_test, y_test, ruta='files/output/metrics.json')
