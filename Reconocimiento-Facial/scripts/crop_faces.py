import os
from PIL import Image
from facenet_pytorch import MTCNN
import torch
import numpy as np
from tqdm import tqdm

RAW_DATA_DIR = 'data/'
CROPPED_DATA_DIR = 'data/cropped/'
INPUT_CLASSES = ['me', 'not_me']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

mtcnn = MTCNN(
    image_size=160,
    margin=0,
    min_face_size=20,
    thresholds=[0.6, 0.7, 0.7],
    factor=0.709,
    post_process=True,
    device=device
)

def crop_faces():
    print("Iniciando detección y recorte facial...")
    
    # Crear directorios de salida
    for class_name in INPUT_CLASSES:
        os.makedirs(os.path.join(CROPPED_DATA_DIR, class_name), exist_ok=True)
    
    # Procesar cada clase de datos
    for class_name in INPUT_CLASSES:
        class_input_dir = os.path.join(RAW_DATA_DIR, class_name)
        class_output_dir = os.path.join(CROPPED_DATA_DIR, class_name)
        
        # Listar archivos de imagen
        image_files = [f for f in os.listdir(class_input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nProcesando {len(image_files)} imágenes para la clase: **{class_name}**")
        
        # Procesar y guardar
        for filename in tqdm(image_files):
            raw_path = os.path.join(class_input_dir, filename)
            cropped_path = os.path.join(class_output_dir, filename)

            try:
                img = Image.open(raw_path)
                
                face_tensor = mtcnn(img, save_path=cropped_path)
                
                if face_tensor is None:
                    print(f"Advertencia: No se detectó un rostro en {filename}. Imagen omitida.")
                    if os.path.exists(cropped_path):
                        os.remove(cropped_path)
                    continue
                    
            except Exception as e:
                print(f"Error procesando {filename}: {e}")
                continue

    print("\nDetección y recorte completado, Los archivos normalizados están en `data/cropped/`.")

if __name__ == '__main__':
    crop_faces()