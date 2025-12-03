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
import os
import json
import gzip
import pickle
import pandas as pd
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    make_scorer,
    precision_score,
    recall_score,
    f1_score
)
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import GridSearchCV




"""
Paso 1

- Renombre la columna "default payment next month" a "default".
- Remueva la columna "ID".
- Elimine los registros con informacion no disponible.
- Para la columna EDUCATION, valores > 4 indican niveles superiores
de educación, agrupe estos valores en la categoría "others".
- Renombre la columna "default payment next month" a "default"
- Remueva la columna "ID".

"""

test_data_pd = pd.read_csv("files/input/test_data.csv.zip", compression='zip')
train_data_pd = pd.read_csv("files/input/train_data.csv.zip", compression='zip')

test_data_pd.rename(columns={"default payment next month": "default"}, inplace=True)
train_data_pd.rename(columns={"default payment next month": "default"}, inplace=True)



test_data_pd.drop(columns=["ID"], inplace=True)
train_data_pd.drop(columns=["ID"], inplace=True)


test_data_pd = test_data_pd.dropna()
train_data_pd = train_data_pd.dropna()


train_data_pd['EDUCATION'] = train_data_pd['EDUCATION'].apply(lambda x: x if x <= 4 else 4)
test_data_pd['EDUCATION'] = test_data_pd['EDUCATION'].apply(lambda x: x if x <= 4 else 4)


"""
Paso 2

Divida los datasets en x_train, y_train, x_test, y_test.
"""


x_train = train_data_pd.drop(columns=["default"])
y_train = train_data_pd["default"]

x_test = test_data_pd.drop(columns=["default"])
y_test = test_data_pd["default"]

"""
Paso 3

Cree un pipeline para el modelo de clasificación. Este pipeline debe
contener las siguientes capas:
- Transforma las variables categoricas usando el método
one-hot-encoding.
- Ajusta un modelo de bosques aleatorios (rando forest).
"""

categorical_columns = ['SEX', 'EDUCATION', 'MARRIAGE']

preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_columns)
    ],
    remainder='passthrough'
)

pipeline = Pipeline(
    steps=[
    ('preprocessor', preprocessor),
    ('rf', RandomForestClassifier(random_state=42))
])

"""
Paso 4.

Optimice los hiperparametros del pipeline usando validación cruzada.
Use 10 splits para la validación cruzada. Use la función de precision
balanceada para medir la precisión del modelo.
"""

parameter_grid = {
    "rf__n_estimators": [100, 200, 500],
    "rf__max_depth": [None, 5, 10],
    "rf__min_samples_split": [2, 5],
    "rf__min_samples_leaf": [1, 2],
}

grid = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring=make_scorer(balanced_accuracy_score),
    cv=10,
    n_jobs=-1,
    verbose=2,
)

grid.fit(x_train, y_train)

"""
Paso 5.
Guarde el modelo (comprimido con gzip) como "files/models/model.pkl.gz".
Recuerde que es posible guardar el modelo comprimido usanzo la libreria gzip.

"""

os.makedirs("files/models", exist_ok=True)

with gzip.open(os.path.join("files", "models", "model.pkl.gz"), "wb") as f:
    pickle.dump(grid, f)

"""
Paso 6
Calcule las metricas de precision, precision balanceada, recall,
y f1-score para los conjuntos de entrenamiento y prueba.
Guardelas en el archivo files/output/metrics.json. Cada fila
del archivo es un diccionario con las metricas de un modelo.
Este diccionario tiene un campo para indicar si es el conjunto
de entrenamiento o prueba. Por ejemplo:
"""


y_pred_train = grid.predict(x_train)
y_pred_test = grid.predict(x_test)

train_metrics = {
    "type": "metrics",
    'dataset': 'train',
    'precision': precision_score(y_train, y_pred_train),
    'balanced_accuracy': balanced_accuracy_score(y_train, y_pred_train),
    'recall': recall_score(y_train, y_pred_train),
    'f1_score': f1_score(y_train, y_pred_train)
}

test_metrics = {
    "type": "metrics",
    'dataset': 'test',
    'precision': precision_score(y_test, y_pred_test),
    'balanced_accuracy': balanced_accuracy_score(y_test, y_pred_test),
    'recall': recall_score(y_test, y_pred_test),
    'f1_score': f1_score(y_test, y_pred_test)
}


"""
Paso 7
Calcule las matrices de confusion para los conjuntos de entrenamiento y
prueba. Guardelas en el archivo files/output/metrics.json. Cada fila
del archivo es un diccionario con las metricas de un modelo.
de entrenamiento o prueba. Por ejemplo:
"""

cm_train = confusion_matrix(y_train, y_pred_train)
cm_test = confusion_matrix(y_test, y_pred_test)

train_cm = {
    "type": "cm_matrix",
    'dataset': 'train',
    'true_0': {
        "predicted_0": int(cm_train[0, 0]),
        "predicted_1": int(cm_train[0, 1])
    },
    'true_1': {
        "predicted_0": int(cm_train[1, 0]),
        "predicted_1": int(cm_train[1, 1])
    }
}

test_cm = {
    "type": "cm_matrix",
    'dataset': 'test',
    'true_0': {
        "predicted_0": int(cm_test[0, 0]),
        "predicted_1": int(cm_test[0, 1])
    },
    'true_1': {
        "predicted_0": int(cm_test[1, 0]),
        "predicted_1": int(cm_test[1, 1])
    }
}

os.makedirs("files/output", exist_ok=True)
with open("files/output/metrics.json", "w", encoding="utf-8") as f:
    f.write(json.dumps(train_metrics) + "\n")
    f.write(json.dumps(test_metrics) + "\n")
    f.write(json.dumps(train_cm) + "\n")
    f.write(json.dumps(test_cm) + "\n")