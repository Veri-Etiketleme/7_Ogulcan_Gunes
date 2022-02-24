import numpy as np
import pandas as pd
import csv

def sigmoid(z):
    # Arguments:
    # z - (numpy matrix) a numpy matrix on which sigmoid function is applied

    A = 1 / (1 + np.exp(-z))
    
    # returns a numpy matrix which has the activation values
    return A;

def relu(z):
    # Arguments:
    # z - (numpy matrix) a numpy matrix on which sigmoid function is applied

    A = z
    A = A * (z > 0)
    
    # returns a numpy matrix which has the activation values
    return A;

def tanh(z):
    # Arguments:
    # z - (numpy matrix) a numpy matrix on which sigmoid function is applied

    A = np.tanh(z)
    
    # returns a numpy matrix which has the activation values
    return A

def forward_propagation(W, b, A_prev, activation_function):
    # Arguments :
    # W, b - (numpy matrix) parameters for current layer
    # A_prev - (numpy matrix) output of last layer
    # activation_function - (string) type of activation function used (sigmoid/relu)   
    
    # z = w1 * a1 + w2 * a2 + .... + b
    Z = np.dot(W, A_prev) + b

    # evaluating A using activation function
    if activation_function == "sigmoid":  
        A = sigmoid(Z)
    elif activation_function == "tanh":
        A = tanh(Z)
    elif activation_function == "relu":
        A = relu(Z)
    else:
        A = Z
    
    # returning output of this layer and cache
    return A

def predict(parameters, X, activation_functions):
    # Arguments :
    # parameters - a dictionary containing the parameters W and b for each layer
    # X - input features
    # activation_functions - (list) Activation function of each layer
    
    # // is used for integer division and we divide by 2 as parameters has dW and db  
    L = len(parameters) // 2;
    
    # finding activation values for each layer and updating caches for each layer
    A_prev = X
    for l in range(L):
        A_prev = forward_propagation(parameters["W" + str(l + 1)], parameters["b" + str(l + 1)], A_prev, activation_functions[l])
    
    #returning y_hat and cache list
    return A_prev

def find_accuracy(parameters, X, y, activation_functions):
    # Arguments:
    # y_hat - predicted output
    # y - actual output
    # activation_functions - (list) Activation function of each layer
    
    # predicting output
    y_hat = predict(parameters, X, activation_functions)
    
    # counting right predictions
    cnt = 0
    
    # assigning number of examples to m
    m = y_hat.shape[1]
    
    # checking prediction for each example
    for i in range(m):
        actual_output = np.argmax(y[:, i].reshape(5, 1), axis = 0) + 1
        
        if activation_functions[len(activation_functions) - 1] == "sigmoid":
            predicted_output = np.argmax(y_hat[:, i].reshape(5, 1), axis = 0) + 1
        else:
            if y_hat[0][i] < 1.5:
                predicted_output = 1
            elif y_hat[0][i] < 2.5:
                predicted_output = 2
            elif y_hat[0][i] < 3.5:
                predicted_output = 3
            elif y_hat[0][i] < 4.5:
                predicted_output = 4
            else:
                predicted_output = 5       
                
        # checking if prediction is right
        if predicted_output == actual_output:
            cnt = cnt + 1
            
    # finding accuracy
    accuracy = cnt * 100 / m
    
    print(accuracy)

def create_test_output(parameters, X, activation_functions):
    # Accuracy :
    # parameters - a dictionary containing the parameters W and b for each layer
    # X - input features
    # activation_functions - (list) Activation function of each layer
   
    # Assigning predicted output to y
    y = predict(parameters, X, activation_functions)
    
    # read as panda dataset from csv files  
    test_data = pd.read_csv('..\Dataset\\test.csv')
    
    # converting to numpy matrix
    test_data = test_data.values
    
    # getting building ids
    test_data = test_data[:, 1]
    
    # assigning number of examples to m
    m = y.shape[1]
    
    # opening submission file
    with open('../csv_files/submission.csv', 'w') as submission_file:
        # using csv file as a dictionary and assigning the columns
        submission_fields = ['building_id', 'damage_grade']
        writer = csv.DictWriter(submission_file, fieldnames = submission_fields)
        
        # writing column titles in csv file
        writer.writeheader()
    
        cnt = [0] * 5
        
        # entering output example by example
        for i in range(m):
            if activation_functions[len(activation_functions) - 1] == "sigmoid":
                predicted_output = np.squeeze(np.argmax(y[:, i].reshape(5, 1), axis = 0) + 1)
                cnt[predicted_output - 1] = cnt[predicted_output - 1] + 1
            else:
                if y[0][i] < 1.5:
                    predicted_output = 1
                elif y[0][i] < 2.5:
                    predicted_output = 2
                elif y[0][i] < 3.5:
                    predicted_output = 3
                elif y[0][i] < 4.5:
                    predicted_output = 4
                else:
                    predicted_output = 5
                    
            if i == 0: 
                print(predicted_output)
        
            writer.writerow({'building_id' : test_data[i], 'damage_grade': 'Grade ' + str(predicted_output)})
                
        print(cnt)

# Support snippets for creating training data
# --------------------------------------------
    
parameters = np.load("../numpy_objects/parameters_trial_8.npy").item()

X_train = np.load("../numpy_objects/X_train.npy")
y_train = np.load("../numpy_objects/y_train.npy")

print("Training Accuracy : ")
find_accuracy(parameters, X_train, y_train, ["relu", "tanh", "relu", "sigmoid"])

X_validation = np.load("../numpy_objects/X_validation.npy")
y_validation = np.load("../numpy_objects/y_validation.npy")

print("Validation Accuracy : ")
find_accuracy(parameters, X_validation, y_validation, ["relu", "tanh", "relu", "sigmoid"])
    
X_test = np.load("../numpy_objects/X_test.npy")
create_test_output(parameters, X_test, ["relu", "tanh", "relu", "sigmoid"])