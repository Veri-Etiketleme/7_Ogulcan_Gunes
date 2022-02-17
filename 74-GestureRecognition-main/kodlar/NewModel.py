from keras.models import Sequential
import cv2
import numpy as np
from keras.preprocessing import image
from keras.models import load_model
from keras.layers import Convolution2D as Conv2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers import Dense, Activation, Flatten

def nothing(x):
    pass

def create_model():
  # Initialising the CNN
  classifier = Sequential()

# Adding first convolutional layer, followed by pooling, and dropout
  classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Dropout(0.25))

# Adding second convolutional layer, followed by pooling, and dropout
  classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Dropout(0.25))

# Adding third convolutional layer, followed by pooling, and dropout
  classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
  classifier.add(MaxPooling2D(pool_size = (2, 2)))
  classifier.add(Dropout(0.25))

# Flattening
  classifier.add(Flatten())

# Full connection
  classifier.add(Dense(units = 128, activation = 'relu'))
  classifier.add(Dense(units = 10, activation = 'softmax'))

# Compiling the CNN
  classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
  return classifier




image_x, image_y = 64,64
model = create_model()
model.load_weights('model.h5')

def predictor():

       img = image.load_img('c2.jpg', target_size=(64, 64))
       img = image.img_to_array(img)
       img = np.expand_dims(img, axis = 0)
       result = model(img)
       
       if result[0][0] == 1:
              return '0'
       elif result[0][1] == 1:
              return '1'
       elif result[0][2] == 1:
              return '2'
       elif result[0][3] == 1:
              return '3'
       elif result[0][4] == 1:
              return '4'
       elif result[0][5] == 1:
              return '5'
       elif result[0][6] == 1:
              return '6'
       elif result[0][7] == 1:
              return '7'
       elif result[0][8] == 1:
              return '8'
       elif result[0][9] == 1:
              return '9'
command = predictor();
print(command);