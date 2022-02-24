import numpy as np

def initialize_parameters(layer_dims, activation_functions):
    # Arguments :
    # layer_dims - (lis)) a L-length list(input layer, hidden layers and output layer) denoting the number of units in each layer 
    # activation_functions - activation function of each layer
    
    # dictionary of parameters W and b
    parameters = {}
    
    # setting L to number of layers
    L = len(layer_dims)
    
    # loading percentage concentration of output
    #percentage_concentration = np.load("../numpy_objects/percentage_concentration.npy").item()
    
    # randomly initializing parameters layer by layer 
    for l in range(1, L):
        W = np.random.randn(layer_dims[l], layer_dims[l - 1])
        b = np.random.randn(layer_dims[l], 1)
        
        # Using He and Xander Initialization
        if activation_functions[l - 1] == "relu" or activation_functions[l - 1] == "sigmoid":
            W = W * np.sqrt(2 / layer_dims[l - 1])
        elif activation_functions[l - 1] == "tanh":
            W = W * np.sqrt(1 / layer_dims[l - 1])
        else:
            W = W * 0.1
        
        # adjusting weights to overcome concentrated output(for sigmoid output)        
        #if output_activation_function == "sigmoid":
            #if l == L - 1:
                #epsilon = np.zeros((5, 1))
                #for grade in range(5):
                    #epsilon[grade] = (100 - percentage_concentration["Grade " + str(grade)]) / 100
                    #epsilon[grade] = 1
                #W = W * epsilon
        
        # Saving the weights in dictionary
        parameters["W" + str(l)] = W
        parameters["b" + str(l)] = b
        
    # returns a dictionary of initialized parameters
    return parameters

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
        
    # creating cache
    cache = (W, b, A_prev, Z)
    
    # returning output of this layer and cache
    return A, cache

def nn_model_forward(parameters, X, activation_functions):
    # Arguments :
    # parameters - a dictionary containing the parameters W and b for each layer
    # X - input features
    # activation_functions - (list) Activation function of each layer
    
     # // is used for integer division and we divide by 2 as parameters has dW and db  
    L = len(parameters) // 2;
    
    # a list of caches
    caches = []
    
    # finding activation values for each layer and updating caches for each layer
    A_prev = X
    for l in range(L):
        A_prev, cache = forward_propagation(parameters["W" + str(l + 1)], parameters["b" + str(l + 1)], A_prev, activation_functions[l])
        caches.append(cache)
    
    #returning y_hat and cache list
    return A_prev, caches
    

def compute_cost(y, y_hat, output_activation_function):
    # Arguments :
    # y - (numpy matrix) actual output
    # y_hat - (numpy matrix) predicted output
    # output_activation_function - (string)activation function of last layer
    
    # assigning number of training examples to m
    m = y.shape[1]
    
    # assigning cost using the loss formula
    if output_activation_function == "sigmoid":
        cost = -np.sum(y * np.log(y_hat) + (1 - y) * np.log(1 - y_hat)) / m
    else:
        cost = np.sum((y_hat - y) ** 2) / (2 * m)
    
    # returns cost
    return cost

def sigmoid_backward(z):
    # Arguments:
    # z - (numpy matrix) a numpy matrix whose g'(z) is to be computed. Here, g is sigmoid function. 
    
    # assigning g(z) to a
    a = sigmoid(z)
    
    # denoting g'(z) by dg
    dg = a * (1 - a)
    
    # return g'(z)
    return dg

def relu_backward(z):
    # Arguments:
    # z - (numpy matrix) a numpy matrix whose g'(z) is to be computed. Here, g is relu function.
    
    # denoting g'(z) by dg
    dg = z > 0
    
    return dg

def tanh_backward(z):
    # Arguments:
    # z - (numpy matrix) a numpy matrix whose g'(z) is to be computed. Here, g is tanh function.
    
    # assigning g(z) to a
    a = tanh(z)
    
    # denoting g'(z) by dg
    dg = 1 - a * a
    
    return dg

def backward_propagation(dA, cache, activation_function):
    # Arguments :
    # dA - (numpy matrix) d(Cost) / d(A)
    # cache - dictionary having W, b, A_prev and Z
    # activation_function - (string) type of activation function used (sigmoid/relu)   
    
    # getting varialbles stored in cache
    W, b, A_prev, Z = cache
    
    # computing dZ based on activation function
    if activation_function == "sigmoid":
        dZ = dA * sigmoid_backward(Z)
    elif activation_function == "tanh":
        dZ = dA * tanh_backward(Z)
    elif activation_function == "relu":
        dZ = dA * relu_backward(Z)
    else:
        dZ = dA
    
    # Computing dW, db and dA_prev    
    dW = np.dot(dZ, A_prev.T)
    db = np.sum(dZ, axis = 1, keepdims = True)
    dA_prev = np.dot(W.T, dZ)
    
    # returns d(Cost) / dW, d(Cost) / db and d(Cost) / d(A_prev)
    return dW, db, dA_prev

def nn_model_backward(caches, y_hat, y, activation_functions):
    # Arguments :
    # caches - list of cache for each layer
    # y_hat - output of last layer
    # y - actual output of last layer
    # activation_functions - (list) Activation function of each layer
    
    # assigning number of layers to L. len + 1 as there is no cache for input layer
    L = len(caches) + 1
    
    # dictionary where grads will be stored
    grads = {}
    
    # assigning number of training examples to m
    m = y_hat.shape[1]
    
    # finding d(cost) / d(y_hat) (depending on activation function in last layer)
    if activation_functions[L - 2] == "sigmoid":
        dA_prev = -(y / y_hat - (1 - y) / (1 - y_hat)) / m;
    else: # considering linear output
        dA_prev = (y_hat - y) / m
    
    # finding dW and db layer by layer in backward direction
    for l in reversed(range(L - 1)):
        grads["dW" + str(l + 1)], grads["db" + str(l + 1)], dA_prev = backward_propagation(dA_prev, caches[l], activation_functions[l])
        
    #returns dictionary containing dW and db for each layer
    return grads

def update_parameters(grads, parameters, learning_rate):
    # Arguments :
    # grads -  dictionary storing dW and db for each layer
    # parameters - dictionary storing W and b for each layer
    # learning_rate - learning rate alpha for gradient descent
    
    # // is used for integer division and we divide by 2 as parameters has dW and db 
    L = len(parameters) // 2
    
    # updating parameters layer by layer
    for l in range(1, L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]
    
    # returning updated parameters
    return parameters

def find_accuracy(y, y_hat, output_activation_function):
    # Arguments:
    # y - actual output
    # y_hat - predicted output
    # output_activation_function - (list) Activation function of last layer
    
    # counting right predictions
    cnt = 0
    
    # assigning number of examples to m
    m = y_hat.shape[1]
    
    # checking prediction for each example
    for i in range(m):
        if output_activation_function == "sigmoid":
            predicted_output = np.argmax(y_hat[:, i].reshape(5, 1), axis = 0) + 1
            actual_output = np.argmax(y[:, i].reshape(5, 1), axis = 0) + 1
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
            if y[0][i] < 1.5:
                actual_output = 1
            elif y[0][i] < 2.5:
                actual_output = 2
            elif y[0][i] < 3.5:
                actual_output = 3
            elif y[0][i] < 4.5:
                actual_output = 4
            else:
                actual_output = 5 
            
        # checking if prediction is right
        if predicted_output == actual_output:
            cnt = cnt + 1
            
    # finding accuracy
    accuracy = cnt * 100 / m
    
    print("Accuracy :" + str(accuracy))

def train_nn_model(layer_dims, activation_functions, X_train, y_train, learning_rate, number_of_iterations):
    # Arguments:
    # layer_dims - (list) Number of layers and number of units in each layer
    # activation_functions - (list) Activation function of each layer
    # X_train - input features for training data
    # y_train - actual output for training data
    # learning_rate - learning rate alpha at which gradient_descent is to be operated
    # number_of_iterations - Number of iterations for which gradient descent is to be operated
    
    # initializing parameters
    parameters = initialize_parameters(layer_dims, activation_functions)
    #parameters = np.load("../numpy_objects/parameters_trial_8.npy").item()
    
    prevcost = 100000000
    
    for i in range(number_of_iterations):
        # running forward propagation to obtain y_hat(predicted output) and caches
        y_hat, caches = nn_model_forward(parameters, X_train, activation_functions)
        
        # checking if there is a y_hat <= 0
        #check = y_hat <= 0
        #if np.sum(check) > 0:
        #    print("y: " + str(y_hat))
            
        #checking if y_hat >= 1
        #check = y_hat >= 1
        #if np.sum(check) > 0:
        #    print("y2 :" + str(y_hat))
        
        # computing cost
        cost = compute_cost(y_train, y_hat, activation_functions[len(activation_functions) - 1])
        
        if prevcost < cost :
            learning_rate = learning_rate / 2
            print("Learning_rate changed : " + str(learning_rate))
        
        prevcost = cost
        
        # running backward propagation to obtain dW and db
        grads = nn_model_backward(caches, y_hat, y_train, activation_functions)
        
        # updating parameters
        parameters = update_parameters(grads, parameters, learning_rate)
        
        # finding accuracy
        if i % 10 == 0:
            find_accuracy(y_train, y_hat, activation_functions[len(activation_functions) - 1])
        
        # printing cost after every iteration
        print("Cost after " + str(i + 1) + " iterations : " + str(cost));
        
        # saving the model parameters after every 10 iterations
        if i % 10 == 0:
            np.save("../numpy_objects/parameters_trial.npy", parameters)
    
# Support snippets for training neural network
# --------------------------------------------
X_train = np.load("../numpy_objects/X_train.npy")
y_train = np.load("../numpy_objects/y_train.npy")

train_nn_model([X_train.shape[0], 70, 70, 70, 70, 5], ["relu", "tanh", "relu", "tanh", "sigmoid"], X_train, y_train, 0.1, 10000)