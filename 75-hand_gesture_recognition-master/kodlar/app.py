import os, sys
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 

import cv2, tensorflow as tf, numpy as np
import logging

try:
    gpus= tf.config.experimental.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(gpus[0], True)
except:
    logging.info('gpu support not set up')

logging.basicConfig(level=logging.INFO)

class main():
    image_output_shape = (640,240)
    classes = ['palm','l','fist','thumb','index','ok','c','down']
    def __init__(self):
        print(sys.argv)
        try:
            self.model = tf.keras.models.load_model('./model/02_iteration_clahe.h5')
        except Exception as e:
            logging.error("Can not load model")
            sys.exit(1)
        if len(sys.argv)<2:
            logging.error("Got no path")
            sys.exit(1)
        self.files = sys.argv[1:]
        self.clahe = cv2.createCLAHE(16,(24,64))
        
    def getter(self):
        for i in self.files:
            if os.path.isfile(i):
                try:
                    photo = cv2.imread(i,cv2.IMREAD_GRAYSCALE)
                except:
                    logging.error(f"Can not read image {i}")
                    continue
                    
                yield photo,i
                
            elif os.path.isdir(i):
                for j in os.listdir(i):
                    try:
                        photo = cv2.imread(os.path.join(i,j),cv2.IMREAD_GRAYSCALE)
                    except:
                        logging.error(f"Can not read image {os.path.join(i,j)}")
                        continue
                    
                    yield photo,os.path.join(i,j)
                    
            else:
                logging.error(f"{i} is not directory or file")
                
    def predict(self):
        for i,j in self.getter():
            clahe_photo = self.clahe.apply(i)
            scaled_photo = cv2.resize(clahe_photo,self.image_output_shape)
            scaled_photo = np.reshape(scaled_photo,(1,)+scaled_photo.shape+(1,))

            output = self.model.predict(scaled_photo)
            logging.info(f"{j} - {self.classes[np.argmax(output)]}")
            
if __name__=='__main__':
    predictor = main()
    logging.info("-------------------")
    logging.info("Starting prediction")
    predictor.predict()
        