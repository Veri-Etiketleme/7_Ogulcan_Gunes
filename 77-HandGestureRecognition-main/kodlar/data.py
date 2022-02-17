import tensorflow as tf
import numpy as np

path = 'HandGestureRecognitionNeuralNetwork/dataset/'


from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(shear_range=0.2, zoom_range=0.4, height_shift_range=20)

train_generator = datagen.flow_from_directory(path, target_size=(256, 256),
    class_mode = 'categorical', shuffle=True)

