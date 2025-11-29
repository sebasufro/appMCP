# evaluate.py
import os
import numpy as np
import pandas as pd
import joblib
import json
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix, accuracy_score
from sklearn.model_selection import train_test_split
from scipy.optimize import brentq
from sklearn.metrics import f1_score
from scipy.interpolate import interp1d

DATA_DIR = 'data/'
MODELS_DIR = 'models/'
REPORTS_DIR = 'reports/'
EMBEDDINGS_FILE = os.path.join(DATA_DIR, 'embeddings.npy')
LABELS_FILE = os.path.join(DATA_DIR, 'labels.csv')
MODEL_PATH = os.path.join(MODELS_DIR, 'model.joblib')
SCALER_PATH = os.path.join(MODELS_DIR, 'scaler.joblib')
CM_PATH = os.path.join(REPORTS_DIR, 'confusion_matrix.png')
OPTIMAL_THRESHOLD_FILE = os.path.join(REPORTS_DIR, 'optimal_threshold.json')

def find_optimal_threshold(fpr, tpr, thresholds):
    fnr = 1 - tpr
    eer_threshold_idx = np.nanargmin(np.abs(fpr - fnr))
    eer_threshold = thresholds[eer_threshold_idx]
    return eer_threshold

def evaluate_model():
    print("Iniciando fase de Evaluación...")
    
    # Cargar Datos y Modelos
    try:
        X = np.load(EMBEDDINGS_FILE)
        y = pd.read_csv(LABELS_FILE)['label'].values
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
    except FileNotFoundError as e:
        print(f"Error: Archivo no encontrado. Error: {e}")
        return
    
    # Re-generar Split de Validación
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_val_scaled = scaler.transform(X_val)
    
    # Calcular Probabilidades
    y_proba = model.predict_proba(X_val_scaled)[:, 1]
    
    # Generar Curva ROC
    fpr, tpr, thresholds_roc = roc_curve(y_val, y_proba)
    roc_auc = auc(fpr, tpr)
    
    # Búsqueda del Umbral Óptimo
    optimal_threshold = find_optimal_threshold(fpr, tpr, thresholds_roc)
    
    # Evaluación con el Umbral Óptimo
    y_pred_optimal = (y_proba >= optimal_threshold).astype(int)
    cm_optimal = confusion_matrix(y_val, y_pred_optimal)
    acc_optimal = accuracy_score(y_val, y_pred_optimal)
    
    print("\n--- Resultados con Umbral Óptimo (EER) ---")
    print(f"Umbral Óptimo: {optimal_threshold:.4f}")
    print(f"Accuracy en validación: {acc_optimal:.4f}")
    print("Matriz de Confusión:\n", cm_optimal)

    # Guardar Umbral y Métricas
    threshold_data = {
        "optimal_threshold": round(optimal_threshold, 4),
        "evaluation_metric": "Equal Error Rate (EER)",
        "validation_accuracy_at_tau": round(acc_optimal, 4),
        "ROC_AUC": round(roc_auc, 4)
    }
    
    os.makedirs(REPORTS_DIR, exist_ok=True)
    with open(OPTIMAL_THRESHOLD_FILE, 'w') as f:
        json.dump(threshold_data, f, indent=4)
    print(f"Umbral óptimo y métricas guardadas en: **{OPTIMAL_THRESHOLD_FILE}**")
    
    # Generar y Guardar Gráficos
    
    # Curva ROC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.scatter(fpr[np.nanargmin(np.abs(fpr - (1 - tpr)))], tpr[np.nanargmin(np.abs(fpr - (1 - tpr)))], marker='o', color='red', s=50, label=f'EER Point (τ={optimal_threshold:.2f})')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR) / Recall')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    
    # Matriz de Confusión
    plt.figure(figsize=(6, 6))
    plt.imshow(cm_optimal, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion Matrix (Optimal $\\tau$)')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['Not Me (0)', 'Me (1)'])
    plt.yticks(tick_marks, ['Not Me (0)', 'Me (1)'])
    
    thresh = cm_optimal.max() / 2.
    for i in range(cm_optimal.shape[0]):
        for j in range(cm_optimal.shape[1]):
            plt.text(j, i, format(cm_optimal[i, j], 'd'),
                     horizontalalignment="center",
                     color="white" if cm_optimal[i, j] > thresh else "black")
            
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()
    
    # Guardar ambos gráficos
    plt.savefig(CM_PATH)
    print(f"Matriz de confusión guardada en: **{CM_PATH}**")
    
    print("\nEvaluación completa")

if __name__ == '__main__':
    os.makedirs(REPORTS_DIR, exist_ok=True)
    evaluate_model()