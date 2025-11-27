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

from pathlib import Path
import zipfile
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_score, balanced_accuracy_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix
import gzip
import pickle
import json

# Crear directorio de salida si no existe
output_dir = Path("files/output")
output_dir.mkdir(parents=True, exist_ok=True)

# Cargar datos
def load_data():
    # train
    with zipfile.ZipFile('files/input/train_data.csv.zip', 'r') as z:
        with z.open('train_default_of_credit_card_clients.csv') as f:
            df_train = pd.read_csv(f)

    # test
    with zipfile.ZipFile('files/input/test_data.csv.zip', 'r') as z:
        with z.open('test_default_of_credit_card_clients.csv') as f:
            df_test = pd.read_csv(f)
    
    return df_train, df_test

# 1. Limpieza de datos
def clean_data(df):
    # Renombrar columna
    df = df.rename(columns={"default payment next month": "default"})
    
    # Remover columna ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])
    
    # Eliminar registros con informacion no disponible
    df = df.dropna()
    
    # Agrupar valores de EDUCATION > 4 en "others"
    df['EDUCATION'] = df['EDUCATION'].apply(lambda x: 4 if x > 4 else x)
    
    return df

# 2. Dividir datasets
def split_data(df_train, df_test):
    x_train = df_train.drop(columns=['default'])
    y_train = df_train['default']
    x_test = df_test.drop(columns=['default'])
    y_test = df_test['default']

    return x_train, y_train, x_test, y_test

# 3. Crear pipeline
def create_pipeline():
    # Identificar columnas categóricas
    categorical_features = ['SEX', 'EDUCATION', 'MARRIAGE']
    
    # Crear el preprocessor para one-hot encoding
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', sparse_output=False, handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough'  # Mantener las demás columnas sin transformar
    )
    
    # Crear el pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=12345))
    ])
    
    return pipeline

# 4. Optimizar hiperparámetros de pipeline con validación cruzada
def optimize_model(pipeline, x_train, y_train):
    # Definir el grid de hiperparámetros a probar
    param_grid = {
        'classifier__n_estimators': [200, 300, 500],
        'classifier__max_depth': [None, 10, 20],
        'classifier__min_samples_split': [2, 5],
        'classifier__min_samples_leaf': [1, 2]
    }
    
    # Crear el GridSearchCV con validación cruzada
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,  # 10 splits para validación cruzada
        scoring='balanced_accuracy',  # Métrica de precisión balanceada
        n_jobs=-1,  # Usar todos los procesadores disponibles
        verbose=1
    )

    # Ajustar el modelo
    grid_search.fit(x_train, y_train)
    
    return grid_search

# 5. Guardar el modelo
def save_model(model):
    # Crear el directorio si no existe
    Path('files/models/model.pkl.gz').parent.mkdir(parents=True, exist_ok=True)
    
    # Guardar el modelo comprimido
    with gzip.open('files/models/model.pkl.gz', 'wb') as f:
        pickle.dump(model, f)
    
    print(f"Modelo guardado en files/models/model.pkl.gz'")

# 6. Calcular métricas y guardar en archivo metrics.json
def calculate_metrics(model, x_train, y_train, x_test, y_test):
    # Crear directorio si no existe
    Path('files/output/metrics.json').parent.mkdir(parents=True, exist_ok=True)
    
    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Métricas para entrenamiento
    train_metrics = {
        'type': 'metrics',
        'dataset': 'train',
        'precision': float(precision_score(y_train, y_train_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_train, y_train_pred)),
        'recall': float(recall_score(y_train, y_train_pred)),
        'f1_score': float(f1_score(y_train, y_train_pred))
    }
    
    # Métricas para prueba
    test_metrics = {
        'type': 'metrics',
        'dataset': 'test',
        'precision': float(precision_score(y_test, y_test_pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y_test, y_test_pred)),
        'recall': float(recall_score(y_test, y_test_pred)),
        'f1_score': float(f1_score(y_test, y_test_pred))
    }
    
    # Guardar métricas
    with open('files/output/metrics.json', 'w') as f:
        f.write(json.dumps(train_metrics) + '\n')
        f.write(json.dumps(test_metrics) + '\n')

    print(f"Métricas guardadas en files/output/metrics.json")

    return train_metrics, test_metrics

# 7. Calcular matrices de confusión
def calculate_confusion_matrices(model, x_train, y_train, x_test, y_test):
    # Predicciones
    y_train_pred = model.predict(x_train)
    y_test_pred = model.predict(x_test)
    
    # Matriz de confusión para entrenamiento
    cm_train = confusion_matrix(y_train, y_train_pred)
    train_cm_dict = {
        'type': 'cm_matrix',
        'dataset': 'train',
        'true_0': {
            'predicted_0': int(cm_train[0, 0]),
            'predicted_1': int(cm_train[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_train[1, 0]),
            'predicted_1': int(cm_train[1, 1])
        }
    }
    
    # Matriz de confusión para prueba
    cm_test = confusion_matrix(y_test, y_test_pred)
    test_cm_dict = {
        'type': 'cm_matrix',
        'dataset': 'test',
        'true_0': {
            'predicted_0': int(cm_test[0, 0]),
            'predicted_1': int(cm_test[0, 1])
        },
        'true_1': {
            'predicted_0': int(cm_test[1, 0]),
            'predicted_1': int(cm_test[1, 1])
        }
    }
    
    # Agregar al archivo existente
    with open('files/output/metrics.json', 'a') as f:
        f.write(json.dumps(train_cm_dict) + '\n')
        f.write(json.dumps(test_cm_dict) + '\n')
    
    print(f"Matrices de confusión guardadas en files/output/metrics.json")

    return train_cm_dict, test_cm_dict

print("0. Cargar datos")
df_train, df_test = load_data()
print("1. Limpiar datos")
df_train_clean = clean_data(df_train)
df_test_clean = clean_data(df_test)
print("2. Dividir datos")
x_train, y_train, x_test, y_test = split_data(df_train_clean, df_test_clean)
print("3,4,5. Crear, optimizar y guardar modelo")
pipeline = create_pipeline()
best_model = optimize_model(pipeline, x_train, y_train)
save_model(best_model)
print("6. Guardar  metricas de precision, precision balanceada, recall y f1-score")
calculate_metrics(best_model,x_train, y_train, x_test, y_test)
print("7. Guardar matrices de confusión")
calculate_confusion_matrices(best_model,x_train, y_train, x_test, y_test)
