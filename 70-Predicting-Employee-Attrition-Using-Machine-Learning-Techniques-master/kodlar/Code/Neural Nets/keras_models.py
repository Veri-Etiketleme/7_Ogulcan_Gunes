import numpy as np
np.random.seed(1)

import plaidml.keras
plaidml.keras.install_backend()

import sys
import argparse
import datetime
import json
import math
import timeit
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.utils import to_categorical
from keras.losses import mean_squared_error
from keras import backend as K

parser = argparse.ArgumentParser(prog='keras_models')
parser.add_argument('modelsfile')
parser.add_argument('trainingfile')
parser.add_argument('testfile')
args = vars(parser.parse_args())

# Read the data
read_train = np.genfromtxt(args['trainingfile'], dtype=float, delimiter=',', skip_header=1)
X_train = read_train[:,0:np.shape(read_train)[1]-1]
y_train = np.transpose(np.matrix(read_train[:,np.shape(read_train)[1]-1]))
y_train = to_categorical(y_train)
read_test = np.genfromtxt(args['testfile'], dtype=float, delimiter=',', skip_header=1)
X_test = read_test[:,0:np.shape(read_test)[1]-1]
y_test = np.transpose(np.matrix(read_test[:,np.shape(read_test)[1]-1]))
y_test = to_categorical(y_test)

# Determine the number of input and output nodes of the neural net
input_size = np.shape(X_train)[1]
output_size = np.shape(y_train)[1]

# Read in the list of models evaluate
with open(args['modelsfile']) as f:
	model_definitions = json.loads(f.read())

for cost_ratio in [ 4.0, 5.0 ]:
	print('cost_ratio {0}'.format(cost_ratio))

	# For every model specified
	for model_definition in model_definitions:
		print('model {0}'.format(model_definition))

		# Create a model
		model = Sequential()

		# Add the layers
		for i in range(len(model_definition['layers']) + 1):

			# The size of the first layer is the number of features, otherwise the size
			# is in the given model definition
			in_size = input_size if i == 0 else model_definition['layers'][i-1]['size']

			# The size of the last layer is the number of values to predict, otherwise
			# the size is in the given model definition
			out_size = output_size if i == len(model_definition['layers']) else model_definition['layers'][i]['size']

			# The activation function is linear for the output layer, otherwise it is
			# in the given model definition
			activation = model_definition['outputactivation'] if i == len(model_definition['layers']) else model_definition['layers'][i]['activation']

			# Add the layer
			model.add(Dense(out_size, input_shape=(in_size,)))
			model.add(Activation(activation))

		# Compile the model with the given optimization method
		#model.compile(loss=lambda y_true, y_pred : my_loss_function(y_true, y_pred, 2, 1), \
		model.compile(loss='categorical_crossentropy', \
						optimizer=model_definition['optimizer'], metrics=['binary_accuracy'])

		class_weight = { 0 : 1.0, 1 : cost_ratio }

		# Fit the model with the given number of epochs
		fit = lambda : \
			model.fit(X_train, y_train, epochs=model_definition['epochs'], class_weight=class_weight,
				batch_size=75, verbose=0, validation_split=0.2)
		seconds = timeit.timeit(fit, number=1)

		(loss, binary_accuracy) = model.evaluate(X_test, y_test)

		# Output the accuracy of the model
		print("Cost ratio:{costratio}	Model:{model}	Loss:{loss}	Accuracy:{binary_accuracy}	Time:{time}".format(
			costratio=cost_ratio, model=model_definition['model'], loss=loss, time=seconds, binary_accuracy=binary_accuracy))
