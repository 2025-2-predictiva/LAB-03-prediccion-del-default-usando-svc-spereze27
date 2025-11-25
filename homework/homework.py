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
# - Descompone la matriz de entrada usando PCA. El PCA usa todas las componentes.
# - Estandariza la matriz de entrada.
# - Selecciona las K columnas mas relevantes de la matrix de entrada.
# - Ajusta una maquina de vectores de soporte (svm).
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
# flake8: noqa: E501
import os
import gzip
import json
import pickle
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    balanced_accuracy_score,
    confusion_matrix
)

# --- CONSTANTES Y CONFIGURACIÓN ---
INPUT_TRAIN = "files/input/train_data.csv.zip"
INPUT_TEST = "files/input/test_data.csv.zip"
MODEL_PATH = "files/models/model.pkl.gz"
METRICS_PATH = "files/output/metrics.json"


def load_and_clean_data(filepath):
    """
    Carga y realiza la limpieza inicial de los datos (Paso 1).
    Aplica las mismas transformaciones para train y test.
    """
    # Carga de datos
    df = pd.read_csv(filepath, compression="zip")

    # Renombrar columna objetivo
    if 'default payment next month' in df.columns:
        df = df.rename(columns={'default payment next month': 'default'})

    # Eliminar columna ID
    if 'ID' in df.columns:
        df = df.drop(columns=['ID'])

    # Eliminar registros con información no disponible (0)
    df = df.loc[df["MARRIAGE"] != 0]
    df = df.loc[df["EDUCATION"] != 0]

    # Agrupar valores > 4 en EDUCATION
    df["EDUCATION"] = df["EDUCATION"].apply(lambda x: 4 if x > 4 else x)

    return df


def split_features_target(df):
    """
    Paso 2: Divide el dataset en X (features) y y (target).
    """
    X = df.drop(columns="default")
    y = df["default"]
    return X, y


def make_pipeline(x_train):
    """
    Paso 3: Crea el pipeline de procesamiento y clasificación.
    """
    categorical_features = ["SEX", "EDUCATION", "MARRIAGE"]
    # Identificar numéricas dinámicamente
    numerical_features = [col for col in x_train.columns if col not in categorical_features]

    # Transformador de columnas
    preprocessor = ColumnTransformer(
        transformers=[
            ("categorical", OneHotEncoder(), categorical_features),
            ("scaler", StandardScaler(), numerical_features),
        ]
    )

    # Definición del pipeline
    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("pca", PCA()),
            ("feature_selection", SelectKBest(score_func=f_classif)),
            ("classifier", SVC(random_state=42)),
        ]
    )
    return pipeline


def create_estimator(pipeline, x_train):
    """
    Paso 4: Configura la búsqueda de hiperparámetros (GridSearchCV).
    """
    # Definición de la malla de parámetros
    param_grid = {
        "pca__n_components": [20, x_train.shape[1] - 2],
        "feature_selection__k": [12],
        "classifier__kernel": ["rbf"],
        "classifier__gamma": [0.1],
    }

    # Configuración del GridSearch
    model = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=10,
        scoring="balanced_accuracy",
        n_jobs=-1,
        refit=True,
    )
    return model


def save_model(model, path):
    """
    Paso 5: Guarda el modelo comprimido con gzip.
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(path), exist_ok=True)

    with gzip.open(path, "wb") as file:
        pickle.dump(model, file)


def calculate_metrics(model, X, y, dataset_name):
    """
    Paso 6: Calcula métricas de rendimiento.
    """
    y_pred = model.predict(X)
    return {
        "type": "metrics",
        "dataset": dataset_name,
        "precision": precision_score(y, y_pred, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "recall": recall_score(y, y_pred, zero_division=0),
        "f1_score": f1_score(y, y_pred, zero_division=0),
    }


def calculate_confusion_matrix_data(model, X, y, dataset_name):
    """
    Paso 7: Calcula y formatea la matriz de confusión.
    """
    y_pred = model.predict(X)
    cm = confusion_matrix(y, y_pred)
    
    return {
        "type": "cm_matrix",
        "dataset": dataset_name,
        "true_0": {
            "predicted_0": int(cm[0, 0]),
            "predicted_1": int(cm[0, 1]),
        },
        "true_1": {
            "predicted_0": int(cm[1, 0]),
            "predicted_1": int(cm[1, 1]),
        },
    }


def main():
    """
    Función orquestadora que ejecuta todos los pasos del flujo de trabajo.
    """
    # 1. Cargar y Limpiar Datos
    print("Cargando y procesando datos...")
    train_df = load_and_clean_data(INPUT_TRAIN)
    test_df = load_and_clean_data(INPUT_TEST)

    # 2. División de Datos
    x_train, y_train = split_features_target(train_df)
    x_test, y_test = split_features_target(test_df)

    # 3. y 4. Construcción y Entrenamiento
    print("Entrenando modelo...")
    pipeline = make_pipeline(x_train)
    estimator = create_estimator(pipeline, x_train)
    estimator.fit(x_train, y_train)

    # 5. Guardar Modelo
    print(f"Guardando modelo en {MODEL_PATH}...")
    save_model(estimator, MODEL_PATH)

    # 6. y 7. Cálculo de Métricas y Matrices
    print("Calculando métricas...")
    metrics_list = []
    
    # Métricas estándar
    metrics_list.append(calculate_metrics(estimator, x_train, y_train, "train"))
    metrics_list.append(calculate_metrics(estimator, x_test, y_test, "test"))
    
    # Matrices de confusión
    metrics_list.append(calculate_confusion_matrix_data(estimator, x_train, y_train, "train"))
    metrics_list.append(calculate_confusion_matrix_data(estimator, x_test, y_test, "test"))

    # Guardar métricas en JSON
    os.makedirs(os.path.dirname(METRICS_PATH), exist_ok=True)
    with open(METRICS_PATH, "w") as f:
        for metric in metrics_list:
            f.write(json.dumps(metric) + "\n")
            
    print("Proceso finalizado exitosamente.")


if __name__ == "__main__":
    main()