from keras.models import Sequential
from PIL import Image, ImageTk
import tkinter as tk
import argparse
import datetime
import cv2
import pygame
import os
import numpy as np

from keras.preprocessing import image
from keras.models import load_model
from keras.layers import Convolution2D as Conv2D, MaxPooling2D, Dropout
from keras.layers.convolutional import Deconv2D as Conv2DTranspose
from keras.layers import Dense, Activation, Flatten
import subprocess

class Application:

    def playAudio(self, a="1"):
        if(int(a)==9):
            subprocess.Popen('C:\\Windows\\System32\\notepad.exe')
        elif(int(a)==8):
            subprocess.Popen('C:\\Windows\\System32\\mspaint.exe')
        elif(int(a)==7):
            subprocess.Popen('C:\\Windows\\System32\\calc.exe')
        elif(int(a)==6):
            subprocess.Popen('C:\\Windows\\explorer.exe')
        elif(int(a)==5):
            subprocess.Popen('C:\\Windows\\System32\\write.exe')
        else:
            pygame.mixer.music.load("audio/"+a+".mp3")
            pygame.mixer.music.play()
        
    def stopAudio(self):
        pygame.mixer.music.stop()

    def pauseAudio(self):
        pygame.mixer.music.pause()

    def unpauseAudio(self):
        pygame.mixer.music.unpause()

    def __init__(self, output_path = "./"):
        self.vs = cv2.VideoCapture(0) 
        pygame.init()
        pygame.mixer.init()
        self.output_path = output_path
        self.current_image = None 

        self.root = tk.Tk()
        self.root.title("Hand shape recognition")
        
        self.root.protocol('WM_DELETE_WINDOW', self.destructor)

        self.panel = tk.Label(self.root) 
        self.panel.pack(padx=10, pady=10)

        btnCapture = tk.Button(self.root, height=1, text="Capture", command=self.take_snapshot)
        btnRecognize = tk.Button(self.root, height=1, text="Recognize", command=self.predictor)
        self.label = tk.Label(self.root, height=1, font="Arial 18 bold", text = "Result")
        btnPlay = tk.Button(self.root, height=1,  text="PLAY", command=self.playAudio)
        btnStop = tk.Button(self.root, height=1,  text="STOP", command=self.stopAudio)
        btnPause = tk.Button(self.root, height=1, text="PAUSE", command=self.pauseAudio)
        btnUnPause = tk.Button(self.root, height=1,  text="UNPAUSE", command=self.unpauseAudio)

        btnCapture.pack(fill="x",  padx=10, pady=10)
        btnRecognize.pack(fill="x",  padx=10, pady=10)
        self.label.pack(fill="x", padx=10, pady=10)
        btnPlay.pack(fill="x", padx=10, pady=10)
        btnStop.pack(fill="x", padx=10, pady=10)
        btnPause.pack(fill="x", padx=10, pady=10)
        btnUnPause.pack(fill="x", padx=10, pady=10)
        
        self.video_loop()

    def video_loop(self):
        ok, frame = self.vs.read()
        if ok:
            cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
            self.current_image = Image.fromarray(cv2image)
            imgtk = ImageTk.PhotoImage(image=self.current_image)
            self.panel.imgtk = imgtk 
            self.panel.config(image=imgtk)  
        self.root.after(30, self.video_loop)

    def take_snapshot(self):
        """ Take snapshot and save it to the file """
        ts = datetime.datetime.now() 
        filename = "{}.jpg".format(ts.strftime("%Y-%m-%d_%H-%M-%S"))  
        p = os.path.join(self.output_path, filename)
        rgb_im = self.current_image.convert('RGB')
        rgb_im.save('myimg.jpg')
        print("[INFO] saved {}".format(filename))

    def create_model(self):
        classifier = Sequential()
        
        classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))

        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))

        classifier.add(Conv2D(32, (3, 3), activation = 'relu'))
        classifier.add(MaxPooling2D(pool_size = (2, 2)))
        classifier.add(Dropout(0.25))

        classifier.add(Flatten())
        classifier.add(Dense(units = 128, activation = 'relu'))
        classifier.add(Dense(units = 10, activation = 'softmax'))

        classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        
        return classifier

    def predictor(self):
        model = self.create_model()
        model.load_weights('model.h5')
        img = image.load_img('myimg.jpg', target_size=(64, 64))
        img = image.img_to_array(img)
        img = np.expand_dims(img, axis = 0)
        result = model(img)
        
        txt = "-"
        if result[0][0] == 1:
            txt = '0'
        elif result[0][1] == 1:
            txt = '1'
        elif result[0][2] == 1:
            txt = '2'
        elif result[0][3] == 1:
            txt = '3'
        elif result[0][4] == 1:
            txt = '4'
        elif result[0][5] == 1:
            txt = '5'
        elif result[0][6] == 1:
            txt = '6'
        elif result[0][7] == 1:
            txt = '7'
        elif result[0][8] == 1:
            txt = '8'
        elif result[0][9] == 1:
            txt = '9'
        print(txt)
        self.label.config(text=txt)
        self.playAudio(txt)

    def destructor(self):
        
        print("[INFO] closing...")
        self.root.destroy()
        self.vs.release() 
        cv2.destroyAllWindows() 

# start the app
print("[INFO] starting hand shape recognition application...")
pba = Application()
pba.root.mainloop()