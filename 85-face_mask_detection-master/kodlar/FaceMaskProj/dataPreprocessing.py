import cv2,os

data_path='C:/Users/KHAAN/OneDrive/Desktop/dataset'
categories=os.listdir(data_path)
labels=[i for i in range(len(categories))]

label_dict=dict(zip(categories,labels)) #empty dictionary

print(label_dict)
print(categories)
print(labels)

img_size = 100
data = []
target = []

for category in categories:
	folder_path = os.path.join(data_path, category)
	img_names = os.listdir(folder_path)
	
	for img_name in img_names:
		img_path = os.path.join(folder_path, img_name)
		img = cv2.imread(img_path)
		
		try:
			gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
			# Coverting the image into gray scale
			resized = cv2.resize(gray, (img_size, img_size))
			# resizing the gray scale into 50x50, since we need a fixed common size for all the images in the dataset
			data.append(resized)
			target.append(label_dict[category])
		# appending the image and the label(categorized) into the list (dataset)
		
		except Exception as e:
			print('Exception:', e)
		# if any exception rasied, the exception will be printed here. And pass to the next image
	
import numpy as np
	
data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)
	
from keras.utils import np_utils
	
new_target = np_utils.to_categorical(target)

np.save('data', data)
np.save('target', new_target)