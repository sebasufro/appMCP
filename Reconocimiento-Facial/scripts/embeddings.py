import os
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import InceptionResnetV1
from PIL import Image
from tqdm import tqdm

CROPPED_DATA_DIR = 'data/cropped/'
OUTPUT_DIR = 'data/'
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, 'embeddings.npy')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'labels.csv')
INPUT_CLASSES = ['me', 'not_me']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def generate_embeddings():
    print("Iniciando generación de embeddings...")
    
    all_embeddings = []
    all_labels = []
    
    # Recorrer las carpetas de imágenes recortadas
    for class_name in INPUT_CLASSES:
        class_dir = os.path.join(CROPPED_DATA_DIR, class_name)
        
        # signar etiqueta numérica
        label = 1 if class_name == 'me' else 0
        
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nGenerando embeddings para la clase: **{class_name}** ({len(image_files)} archivos)")
        
        # Procesar y generar embeddings
        for filename in tqdm(image_files):
            file_path = os.path.join(class_dir, filename)
            
            try:
                # Cargar la imagen
                img = Image.open(file_path).convert('RGB')
                
                # Convertir a tensor y preprocesar
                img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                
                # Desactivar cálculo de gradientes para solo inferencia
                with torch.no_grad():
                    embedding = resnet(img_tensor).cpu().numpy().flatten()
                
                all_embeddings.append(embedding)
                all_labels.append(label)
                
            except Exception as e:
                print(f"Error generando embedding para {filename}: {e}")
                continue

    # Guardar los resultados como un array numpy
    embeddings_array = np.array(all_embeddings)
    np.save(EMBEDDINGS_FILE, embeddings_array)
    print(f"\nEmbeddings guardados en: **{EMBEDDINGS_FILE}** ({embeddings_array.shape})")
    
    # Guardar las etiquetas como un archivo CSV
    labels_df = pd.DataFrame({'filename': [f for f in os.listdir(os.path.join(CROPPED_DATA_DIR, 'me'))] + 
                                         [f for f in os.listdir(os.path.join(CROPPED_DATA_DIR, 'not_me')) if os.path.exists(os.path.join(CROPPED_DATA_DIR, 'not_me', f))],
                              'label': all_labels})
    labels_df.to_csv(LABELS_FILE, index=False)
    print(f"Etiquetas guardadas en: **{LABELS_FILE}**")

    print("\nGeneración de embeddings completada")

if __name__ == '__main__':
    generate_embeddings()