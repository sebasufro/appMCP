import os
import time
import json
import joblib
import numpy as np
from io import BytesIO
from flask import Flask, request, jsonify
from facenet_pytorch import MTCNN, InceptionResnetV1
from PIL import Image
import torch
from dotenv import load_dotenv

load_dotenv(os.path.join(os.path.dirname(__file__), '..', '.env.example'))

app = Flask(__name__)

THRESHOLD = float(os.getenv('THRESHOLD'))
MODEL_VERSION = os.getenv('MODEL_VERSION')
MAX_MB = int(os.getenv('MAX_MB'))
MAX_BYTES = MAX_MB * 1024 * 1024

BASE_DIR = os.path.join(os.path.dirname(__file__), '..')
MODEL_PATH = os.path.join(BASE_DIR, os.getenv('MODEL_PATH', 'models/model.joblib'))
SCALER_PATH = os.path.join(BASE_DIR, os.getenv('SCALER_PATH', 'models/scaler.joblib'))

# Dispositivo para PyTorch
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Inicializar MTCNN y FaceNet
print(f"Cargando FaceNet en {DEVICE}...")
mtcnn = MTCNN(image_size=160, margin=0, device=DEVICE)
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(DEVICE)

# Cargar el clasificador y el escalador de scikit-learn
try:
    classifier = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    print("Clasificador y escalador cargados exitosamente.")
except Exception as e:
    print(f"Error al cargar modelos: {e}")
    classifier = None
    scaler = None

@app.route('/status', methods=['GET'])
def status():
    return jsonify({
        "status": "ok",
        "model_loaded": classifier is not None,
        "device": str(DEVICE)
    }), 200

@app.route('/verify', methods=['POST'])
def verify():
    start_time = time.time()

    log_data = {
        "endpoint": "/verify",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "result": "ERROR",
        "error_message": None,
        "timing_ms": 0.0,
        "score": None,
        "is_me": None,
        "input_size_bytes": 0
    }

    # Validación de Entrada
    if 'image' not in request.files:
        return jsonify({"error": "campo 'image' no encontrado"}), 400

    img_file = request.files['image']
    
    # Validar tipo de archivo
    if img_file.mimetype not in ['image/jpeg', 'image/png', 'image/jpg']:
        return jsonify({"error": "solo image/jpeg, image/png o image/jpg"}), 400
    
    # Validar tamaño del archivo
    img_bytes = img_file.read()
    log_data["input_size_bytes"] = len(img_bytes)

    if len(img_bytes) > MAX_BYTES:
        log_data["error_message"] = f"Archivo demasiado grande. Máximo {MAX_MB}MB"
        log_data["result"] = "RECHAZO_TAMAÑO"
        log_data["timing_ms"] = round((time.time() - start_time) * 1000, 2)
        print(json.dumps(log_data))
        return jsonify({"error": f"Archivo demasiado grande. Máximo {MAX_MB}MB"}), 413
    
    try:
        img = Image.open(BytesIO(img_bytes)).convert('RGB')
    except Exception:
        log_data["error_message"] = "No se pudo procesar la imagen"
        log_data["result"] = "ERROR_PROCESAMIENTO_IMAGEN"
        log_data["timing_ms"] = round((time.time() - start_time) * 1000, 2)
        print(json.dumps(log_data))
        return jsonify({"error": "No se pudo procesar la imagen"}), 400

    # Preprocesamiento
    try:
        face_tensor = mtcnn(img, return_prob=False, save_path=None)
        
        if face_tensor is None:
            log_data["error_message"] = "No se pudo detectar un rostro en la imagen"
            log_data["result"] = "RECHAZO_NO_ROSTRO"
            log_data["timing_ms"] = round((time.time() - start_time) * 1000, 2)
            print(json.dumps(log_data))
            return jsonify({"error": "No se pudo detectar un rostro en la imagen"}), 400

        face_tensor = face_tensor.to(DEVICE).unsqueeze(0)

        with torch.no_grad():
            embedding = resnet(face_tensor).cpu().numpy()

        # Clasificación
        embedding_scaled = scaler.transform(embedding)
        score = classifier.predict_proba(embedding_scaled)[0, 1]
        
        # Decisión y Umbral de Seguridad
        is_me = bool(score >= THRESHOLD)
        
        # Respuesta
        log_data["result"] = "VERIFICADO" if is_me else "RECHAZO_PUNTUACION"
        log_data["score"] = round(score, 4)
        log_data["is_me"] = is_me
        log_data["timing_ms"] = round((time.time() - start_time) * 1000, 2)
        
        print(json.dumps(log_data))

        response = {
            "model_version": MODEL_VERSION,
            "is_me": is_me,
            "score": round(score, 4),
            "threshold": THRESHOLD,
            "timing_ms": round((time.time() - start_time) * 1000, 2)
        }
        
        return jsonify(response), 200

    except Exception as e:
        # Manejo de excepciones internas
        log_data["error_message"] = f"Error interno: {str(e)}"
        log_data["result"] = "ERROR_INTERNO"
        log_data["timing_ms"] = round((time.time() - start_time) * 1000, 2)
        print(json.dumps(log_data))
        return jsonify({"error": f"Error interno del servidor: {str(e)}"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=os.getenv('DEBUG', 'True') == 'True')