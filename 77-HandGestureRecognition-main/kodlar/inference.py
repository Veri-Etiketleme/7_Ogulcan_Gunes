import tflite_runtime.interpreter as tflite
from PIL import Image
import time
import numpy as np
import argparse

def parseArgs():
	parser = argparse.ArgumentParser()
	parser.add_argument('-path', dtype = str, default = 'output.jpg',
			help = 'Set the image path for inference')
	args = parser.parse_args()
	return args

model_path = '/home/pi/Documents/hands/HandGestureRecognition/foo.tflite'
filepath = 'left.jpg'

def run_inference():
	interpreter = tflite.Interpreter(model_path = model_path)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()

	floating_model = input_details[0]['dtype'] == np.float32

	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]

	img = Image.open(filepath).resize((width, height))
	input_data = np.expand_dims(img, axis = 0)

	if floating_model:
		input_data = np.float32(input_data)
	interpreter.set_tensor(input_details[0]['index'], input_data)

	start_time = time.time()
	interpreter.invoke()
	end_time = time.time()

	output_data = interpreter.get_tensor(output_details[0]['index'])
	results = np.squeeze(output_data)

	d = {0: "left", 1: "right", 2: "palm", 3: "peace"}

	index = np.argmax(results)
	print(d[index])
	print('time: {:3f}ms'.format((end_time - start_time)*1000))


def main():
	args = parseArgs()
	run_inference(args.path)



if __name__ == '__main__':
	main()
