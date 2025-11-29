# Modelo de Reconocimiento Facial

Este proyecto implementa un sistema de verificación facial binaria, identificando al usuario del resto

## Dependencias Utilizadas:

- facenet-pytorch
- scikit-learn
- numpy
- matplotlib
- python-dotenv
- Flask
- gunicorn
- joblib
- Pillow

## Ejecución del codigo

### 1. Creación del entorno virtual
Para este proyecto, se tiene que crear un entorno virtual con ````python -m venv .venv````.
Una vez creado, se tiene que instalar las dependencias que se encuentran en requirements.txt con ````pip install -r requirements.txt````

### 2. Recorte de imagenes
Se ejecuta el script de crop_faces con ````python scripts/crop_faces.py```` para que las imagenes tengan un tamaño aceptable para luego clasificarlas.

### 3. Crear los embeddings
Se ejecuta el archivo ````python scripts/embeddings.py```` para organizar las imagenes recortadas para su uso en el entrenamiento del modelo, clasificandolas en dos clases: "me" y "not-me".

### 4. Entrenar y Evaluar el modelo
El modelo se entrena ejecutando ````python train.py````, el cual utiliza LogisticRegression para clasificar los embeddings creados anteriormente para distinguir las clases correspondientes a las imagenes.

Una vez entrenado, se puede ejecutar ````python evaluate.py```` para generar las metricas y graficos del rendimiento del modelo.

### 5. Uso del modelo
Se puede comprobar el uso del modelo, ejecutandolo con ````python api/app.py````, y utilizando un software de prueba de endpoints como Insomnia, o usando curl en la consola.

Con curl, se puede usar de esta forma: ````curl -F "image=@muestra/selfie.jpg" http://localhost:5000/verify````, y se devuelve un json indicando la clase, la exactitud, y el tiempo que se demora en analizar la imagen.