import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
import joblib
import json

DATA_DIR = 'data/'
MODELS_DIR = 'models/'
REPORTS_DIR = 'reports/'
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'embeddings.npy')
LABELS_FILE = os.path.join(DATA_DIR, 'labels.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
METRICS_PATH = os.path.join(REPORTS_DIR, 'metrics.json')

def train_classifier():
    print("Iniciando fase de entrenamiento")
    
    # Cargar Datos
    try:
        X = np.load(EMBEDDINGS_FILE)
        y_df = pd.read_csv(LABELS_FILE)
        y = y_df['label'].values
    except FileNotFoundError:
        print(f"Error: Archivos de datos no encontrados. Buscando en: {EMBEDDINGS_FILE}")
        return
    
    print(f"Datos cargados. Embeddings shape: {X.shape}, Etiquetas shape: {y.shape}")
    
    # Dividir datos en entrenamiento y validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"Split: {len(X_train)} muestras para entrenamiento, {len(X_val)} para validación.")
    
    # Normalización/Escalado
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    
    # Inicializar y Entrenar el Clasificador
    model = LogisticRegression(max_iter=200, random_state=42, class_weight='balanced')
    print("Entrenando Logistic Regression...")
    model.fit(X_train_scaled, y_train)
    
    # Evaluación simple en el conjunto de validación
    y_pred = model.predict(X_val_scaled)
    y_proba = model.predict_proba(X_val_scaled)
    
    # Métricas
    accuracy = accuracy_score(y_val, y_pred)
    try:
        auc = roc_auc_score(y_val, y_proba, multi_class='ovr')
    except ValueError:
        # Fallback for binary classification if only 2 classes are present in validation set
        if y_proba.shape[1] == 2:
             auc = roc_auc_score(y_val, y_proba[:, 1])
        else:
             auc = 0.0 # Should not happen with OVR but safe fallback

    cm = confusion_matrix(y_val, y_pred).tolist()
    
    metrics = {
        "validation_accuracy": round(accuracy, 4),
        "validation_auc": round(auc, 4),
        "validation_confusion_matrix": cm,
        "model_type": "LogisticRegression",
        "n_features": X.shape[1],
        "n_samples": X.shape[0]
    }
    
    # Guardar Modelo, Escalador y Métricas
    os.makedirs(MODELS_DIR, exist_ok=True)
    os.makedirs(REPORTS_DIR, exist_ok=True)
    
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    
    with open(METRICS_PATH, 'w') as f:
        json.dump(metrics, f, indent=4)
    
    print("\n--- Resultados de Validación Rápida (Entrenamiento) ---")
    print(f"Accuracy: {metrics['validation_accuracy']}")
    print(f"AUC: {metrics['validation_auc']}")
    print(f"Modelo guardado en: **{MODEL_PATH}**")
    print(f"Métricas guardadas en: **{METRICS_PATH}**")
    print("\n¡Entrenamiento completado!")

if __name__ == '__main__':
    train_classifier()