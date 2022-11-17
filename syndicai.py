class PythonPredictor:
#Descomprimimos el archivo rar
!unrar x Perros_y_Gatos.rar

#importamos librerias
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Definimos los directorios
base_dir = 'Perros y Gatos'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

train_gatos_dir = os.path.join(train_dir, 'gatos')
train_perros_dir = os.path.join(train_dir, 'perros')

validation_gatos_dir = os.path.join(validation_dir, 'gatos')
validation_perros_dir = os.path.join(validation_dir, 'perros')

test_gatos_dir = os.path.join(test_dir, 'gatos')
test_perros_dir = os.path.join(test_dir, 'perros')

len(os.listdir(train_perros_dir))

#importamos librerias de tensorflow
import tensorflow
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

#Procesamos las imagenes
from keras.preprocessing.image import ImageDataGenerator
train_data = ImageDataGenerator(rescale=1. /255)
val_data = ImageDataGenerator(rescale=1. /255)
test_data = ImageDataGenerator(rescale=1. /255)

#from keras.engine import training
#Cargamos los datos de las imagenes
training_set = train_data.flow_from_directory(
    train_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)
validation_set = val_data.flow_from_directory(
    validation_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)
test_set = test_data.flow_from_directory(
    test_dir,
    target_size=(64,64),
    batch_size=20,
    class_mode='binary'
)

#from keras.engine import training
training_set.class_indices

#Visualizamos algunas imagenes
from keras.utils import load_img
fnames = [os.path.join(train_gatos_dir, fname) for
  fname in os.listdir(train_gatos_dir)]
img_path = fnames[666]
img = load_img(img_path, target_size=(150,150))

plt.figure()
imgplot = plt.imshow(img)
plt.show()

#construimos la red convolucional o algo asi
red = Sequential()
red.add(Conv2D(32, (3,3), input_shape=(64,64,3), activation='relu'))
red.add(MaxPooling2D(2,2))
red.add(Conv2D(64, (3,3), activation='relu'))
red.add(MaxPooling2D(2,2))
red.add(Conv2D(128, (3,3), activation='relu'))
red.add(MaxPooling2D(2,2))
red.add(Conv2D(128, (3,3), activation='relu'))
red.add(MaxPooling2D(2,2))
red.add(Flatten())
red.add(Dropout(0.5))
red.add(Dense(units=512, activation='relu'))
red.add(Dense(units=1, activation='sigmoid'))

red.summary()

from keras import optimizers

red.compile(loss='binary_crossentropy',
            optimizer=optimizers.RMSprop(learning_rate=1e-4),
            metrics=['accuracy', 'mse'])

#from IPython.core import history
#Entrenamiento
history = red.fit(training_set,
                  steps_per_epoch=100,
                  epochs=30,
                  batch_size=50,
                  validation_data=validation_set,
                  validation_steps=50)

plt.plot(history.history['loss'], label='loss')
plt.plot(history.history['val_loss'], label='val_loss')
plt.legend()
plt.show()

plt.plot(history.history['mse'], label='mse')
plt.plot(history.history['val_mse'], label='val_mse')
plt.legend()
plt.show()

plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.legend()
plt.show()

predict_set = red.predict(test_set)

from sklearn.metrics import confusion_matrix, accuracy_score

red.evaluate(test_set)

#guardamos el modelo de red neuronal
red.save("RedCNN_PerrosyGatos.h5")

nueva_red = keras.models.load_model("RedCNN_PerrosyGatos.h5")

#Predecimos una sola image
path = "Perros y Gatos/validation/perros/dog.1004.jpg"
from keras.utils import load_img, img_to_array
img_tensor = img_to_array(img)
img_tensor = np.expand_dims(img_tensor, axis=0)
img_tensor /= 255
print(img_tensor.shape)

#mostramos la imagen cargada
plt.imshow(img_tensor[0])
plt.show()

animal = nueva_red.predict(img_tensor)

if np.round(animal[0][0])==1;
print("Perro")
else:
  print("Gato")
