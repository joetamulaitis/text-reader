#Create a CNN to recognize handwritten digits
import emnist
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

import numpy as np
import matplotlib.pyplot as plt

from emnist import extract_training_samples
from emnist import list_datasets
from emnist import extract_test_samples

#Extract the train and test data from EMNIST dataset
train_data, train_labels = extract_training_samples('byclass')
test_data, test_labels = extract_test_samples('byclass')

#Normalize pixel values to be 0-1
train_data = tf.keras.utils.normalize(train_data, axis = 1)
test_data = tf.keras.utils.normalize(test_data, axis = 1)

train_data = np.expand_dims(train_data, axis=3)
test_data = np.expand_dims(test_data, axis=3)

#Data Augmentation to Add Variation to Dataset
rotation_range = 10
width_shift = 0.20
height_shift = 0.20

train_gen = ImageDataGenerator(rotation_range = rotation_range,
                             width_shift_range = width_shift,
                             height_shift_range = height_shift)

train_gen.fit(train_data.reshape(train_data.shape[0], 28, 28, 1))


val_datagen = ImageDataGenerator(rotation_range = rotation_range,
                             width_shift_range = width_shift,
                             height_shift_range = height_shift)
val_datagen.fit(test_data.reshape(test_data.shape[0], 28, 28, 1))

#Create Model
model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1), padding='same'),
    tf.keras.layers.MaxPooling2D((2, 2)),
    
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Conv2D(128, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D(2,2),

    tf.keras.layers.Dropout(0.2),
    
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(256, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(62, activation = 'softmax')
])

#Compile and Train Model
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=['accuracy'])

model.fit(train_data, train_labels, epochs=5)