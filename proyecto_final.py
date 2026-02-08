# Se importan librerias
import os
import numpy as np
from PIL import Image

EPOCHS = 250

BATCH_SIZE = 32 

# Ruta base de tu dataset
ruta_dataset = r"C:\Users\Admin\Downloads\archive (2)"  # Carpeta con dataset

# Tamaño de las imágenes
IMG_SIZE = (244, 244)

def cargar_dataset(ruta_split):
    """
    Carga imágenes y etiquetas desde una subcarpeta (train/test/val)
    """
    X = []
    y = []
    clases = sorted(os.listdir(ruta_split))  # nombres de carpetas / clases
    class_to_idx = {clase: i for i, clase in enumerate(clases)}

    for clase in clases:
        ruta_clase = os.path.join(ruta_split, clase)

        if not os.path.isdir(ruta_clase):
            continue

        for archivo in os.listdir(ruta_clase):
            if archivo.lower().endswith((".jpg", ".jpeg", ".png")):
                path_img = os.path.join(ruta_clase, archivo)

                # Abrir imagen
                img = Image.open(path_img).convert("RGB")
                img = img.resize(IMG_SIZE)

                # Convertir a array numpy
                img_np = np.array(img)

                X.append(img_np)
                y.append(class_to_idx[clase])

    # Convertir listas a arrays
    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)

    return X, y, class_to_idx

# Cargar las carpetas de train, test, val
X_carpeta_train, y_carpeta_train, class_map = cargar_dataset(os.path.join(ruta_dataset, "train"))
X_carpeta_test, y_carpeta_test, _ = cargar_dataset(os.path.join(ruta_dataset, "test"))
X_carpeta_val, y_carpeta_val, _ = cargar_dataset(os.path.join(ruta_dataset, "val"))

# Normalizamos
X_carpeta_train = X_carpeta_train / 255.0
X_carpeta_test = X_carpeta_test / 255.0
X_carpeta_val = X_carpeta_val / 255.0

# Concatenamos los datos
X = np.concatenate((X_carpeta_train, X_carpeta_test, X_carpeta_val), axis=0)
y = np.concatenate((y_carpeta_train, y_carpeta_test, y_carpeta_val), axis=0)

# Dejamos solo las imagénes que no se repiten
X_reshape = X.reshape(X.shape[0], -1)
_, indices_unicos = np.unique(X_reshape, axis=0, return_index=True)
indices_unicos = np.sort(indices_unicos)
X_unicos = X[indices_unicos]
y_unicos = y[indices_unicos]

from tensorflow.keras.utils import to_categorical
y_onehot = to_categorical(y_unicos, num_classes=6)                                      # Se hace para perder "cercanía" entre las clases

# Por último, se divide el dataset en train, test y val
from sklearn.model_selection import train_test_split

# Primero dividimos en train y test
X_train, X_temp, y_train, y_temp = train_test_split(X_unicos, y_onehot, test_size=0.30, random_state=42, shuffle=True)

# Luego dividimos el temporal en val y test
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.50, random_state=42, shuffle=True)

"""
# Información del dataset
print("Clases:", class_map)                                                      Muestra a qué clase se refiere cada número de etiqueta 
# print(f"La forma de X antes de eliminar duplicados: {X.shape}")
# print(f"La forma de X después de eliminar duplicados: {X_unicos.shape}")
# print(y.shape)

import matplotlib.pyplot as plt

print(y[500])                                                                    Imprime la etiqueta de la imagen a mostrar
# Muestra una imagen 
plt.imshow(X[500])
plt.axis('off')   # quita los ejes
plt.show()
"""
import tensorflow as tf
from tensorflow.keras import layers

tf.get_logger().setLevel('ERROR')

data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),    # voltea horizontal
    layers.RandomRotation(0.2),         # rota ±20%
    layers.RandomZoom(0.15),             # zoom in/out
    layers.RandomContrast(0.2),         # cambia contraste
])

model = tf.keras.Sequential([
    data_augmentation,
    tf.keras.layers.Conv2D(64, (4,4), strides=(2,2), activation='relu', input_shape=(244, 244, 3)),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(64, (3,3), strides=(2,2), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Conv2D(32, (3,3), strides=(2,2), activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dropout(0.3),
    tf.keras.layers.Dense(6, activation='softmax')
])

from keras.optimizers import Adam

# Optimizador
optimizer = Adam(learning_rate=0.0001, beta_1=0.9, beta_2=0.999)

# Compila el modelo
model.compile(optimizer = optimizer, loss = 'categorical_crossentropy', metrics = ['accuracy'])

model.build((None, 244, 244, 3))
model.summary()
history = model.fit(X_train, y_train, validation_data = (X_val, y_val), epochs=EPOCHS, batch_size=BATCH_SIZE, verbose=2, shuffle=True)

# Obtenemos la historia de entrenamiento (matriz con columnas con val y train acc y val y train loss, con epochs filas)
history_dict = history.history

epochs_range = range(1, len(history_dict['loss']) + 1)

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history_dict['loss'], label='Pérdida entrenamiento')
plt.plot(epochs_range, history_dict['val_loss'], label='Pérdida validación')
plt.xlabel('Época')
plt.ylabel('Pérdida')
plt.title('Pérdida de entrenamiento y validación')
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(8, 5))
plt.plot(epochs_range, history_dict['accuracy'], label='Precisión entrenamiento')
plt.plot(epochs_range, history_dict['val_accuracy'], label='Precisión Validación')
plt.xlabel('Época')
plt.ylabel('Precisión')
plt.title('Precisión de entrenamiento y validación')
plt.legend()
plt.grid(True)
plt.show()
