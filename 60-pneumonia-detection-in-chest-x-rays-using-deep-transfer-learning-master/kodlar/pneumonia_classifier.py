#import libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import os
from tqdm import tqdm
import skimage
from keras.callbacks import Callback, EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
from keras.layers import Dense
from keras.models import Model
from keras.applications.resnet50 import ResNet50
from keras.applications.vgg16 import VGG16
from keras.applications.inception_v3 import InceptionV3
from keras.applications.xception import Xception
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from sklearn.utils import class_weight
import keras
from keras.layers import Flatten

#load in data (numpy arrays)
X_train=np.load("Datasets/X_train.npy")
y_trainHot= np.load("Datasets/y_trainHot.npy")
X_test=np.load("Datasets/X_test.npy")
y_testHot= np.load("Datasets/y_testHot.npy")


class_weight1 = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
weight_path = 'imagenet'


#import pretrained networks
pretrained_model_1 = VGG16(weights = weight_path, include_top=False, input_shape=(200, 200, 3))
pretrained_model_2 = InceptionV3(weights = weight_path, include_top=False, input_shape=(200, 200, 3))
pretrained_model_3 = ResNet50(weights = weight_path, include_top=False, input_shape=(200, 200, 3))
pretrained_model_4 = InceptionResNetV2(weights = weight_path, include_top=False, input_shape=(200, 200, 3))
pretrained_model_5 = Xception(weights = weight_path, include_top=False, input_shape=(200, 200, 3))
optimizer1 = keras.optimizers.Adam(lr=0.0001)

#function to plot errors per epoch and accuracy per epoch
def model_plots(history):
    plt.figure(figsize=(8,8))
    plt.subplot(1,2,1)
    plt.plot(history.history['acc'])
    plt.plot(history.history['val_acc'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.figure(1)
    plt.subplot(1,2,2)
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')
 

 #function that retrains final layer of a pretrained network on pneumonia data
def retrain(xtrain,ytrain,xtest,ytest,pretrainedmodel,pretrainedweights,classweight,numclasses,epochs,optimizer,labels):
    base_model = pretrainedmodel # Topless
    # Add top layer
    x = base_model.output
    x = Flatten()(x)
    predictions = Dense(numclasses, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)


    # Train top layer only, freeze previous weights
    for layer in base_model.layers:
        layer.trainable = False #only train final layer
    model.compile(loss='categorical_crossentropy', 
                  optimizer=optimizer, 
                  metrics=['accuracy'])
    callbacks_list = [keras.callbacks.EarlyStopping(monitor='val_acc', patience=3, verbose=1)]
    model.summary()


    # train model on new pneumonia data
    history = model.fit(xtrain,ytrain, epochs=epochs, class_weight=classweight, validation_data=(xtest,ytest), verbose=1)
    
    # Evaluate model
    score = model.evaluate(xtest,ytest, verbose=0)
    print('\nKeras CNN - accuracy:', score[1], '\n')

    y_pred = model.predict(xtest) #predictions

    Y_pred_classes = np.argmax(y_pred,axis = 1) 
    Y_true = np.argmax(ytest,axis = 1) 


    model_plots(history)
    plt.show()
    return model




#function trains one of the pretrained models (only last layer) on train set and evaluates model on test set, creates error and accuracy per epoch plot
retrain(X_train, y_trainHot, X_test, y_testHot,pretrained_model_1,weight_path,class_weight1,2,20,optimizer1,{0: 'No Pneumonia', 1: 'Yes Pneumonia'})
