import os
import json
import torch
import numpy as np
import pandas as pd
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from PIL import Image
from tqdm import tqdm

CROPPED_DATA_DIR = 'data/cropped/'
OUTPUT_DIR = 'data/'
EMBEDDINGS_FILE = os.path.join(OUTPUT_DIR, 'embeddings.npy')
LABELS_FILE = os.path.join(OUTPUT_DIR, 'labels.csv')
INPUT_CLASSES = ['Yo', 'Robin', 'Francisco']

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print(f'Usando dispositivo: {device}')

resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

def generate_embeddings():
    print("Iniciando generación de embeddings...")
    
    all_embeddings = []
    all_labels = []
    
    # Recorrer las carpetas de imágenes recortadas
    all_filenames = []
    
    for i, class_name in enumerate(INPUT_CLASSES):
        class_dir = os.path.join(CROPPED_DATA_DIR, class_name)
        
        # Asignar etiqueta numérica basada en el índice de la clase
        label = i
        
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        print(f"\nGenerando embeddings para la clase: **{class_name}** (label: {label}) ({len(image_files)} archivos)")
        
        # Procesar y generar embeddings
        for filename in tqdm(image_files):
            file_path = os.path.join(class_dir, filename)
            
            try:
                # Cargar la imagen
                img = Image.open(file_path).convert('RGB')
                
                # Resize a 160x160 (entrada esperada por InceptionResnetV1)
                img = img.resize((160, 160))
                
                # Convertir a tensor y preprocesar (standardization)
                img_tensor = torch.as_tensor(np.array(img), dtype=torch.float32).permute(2, 0, 1).unsqueeze(0).to(device)
                img_tensor = fixed_image_standardization(img_tensor)
                
                # Desactivar cálculo de gradientes para solo inferencia
                with torch.no_grad():
                    embedding = resnet(img_tensor).cpu().numpy().flatten()
                
                all_embeddings.append(embedding)
                all_labels.append(label)
                all_filenames.append(filename)
                
            except Exception as e:
                print(f"Error generando embedding para {filename}: {e}")
                continue

    # Guardar los resultados como un array numpy
    embeddings_array = np.array(all_embeddings)
    np.save(EMBEDDINGS_FILE, embeddings_array)
    print(f"\nEmbeddings guardados en: **{EMBEDDINGS_FILE}** ({embeddings_array.shape})")
    
    # Guardar las etiquetas como un archivo CSV
    labels_df = pd.DataFrame({'filename': all_filenames,
                              'label': all_labels})
    labels_df.to_csv(LABELS_FILE, index=False)
    print(f"Etiquetas guardadas en: **{LABELS_FILE}**")

    # Guardar el mapeo de clases
    class_mapping = {i: name for i, name in enumerate(INPUT_CLASSES)}
    mapping_path = os.path.join(OUTPUT_DIR, 'classes.json')
    with open(mapping_path, 'w') as f:
        json.dump(class_mapping, f, indent=4)
    print(f"Mapeo de clases guardado en: **{mapping_path}**")

    print("\nGeneración de embeddings completada")

if __name__ == '__main__':
    generate_embeddings()