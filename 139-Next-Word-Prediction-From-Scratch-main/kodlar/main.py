import numpy as np
from matplotlib import pyplot as plt
from sklearn.utils import shuffle
import pickle
from network import NeuralNetwork
import eval
import time

"""
Main Class
...

Loads necessary files,
Initializes the Network,
Runs the training loop,
Saves the model after training

"""

def convert_one_hot_to_index(one_hot_vector):
    index = 0
    for i in range(len(one_hot_vector)):
        if one_hot_vector[i] == 1:
            index = i
    return index

def convert_one_hot(word_index):
    one_hot_representation = np.zeros(250)
    one_hot_representation[word_index] = 1
    return one_hot_representation

def convert_one_hot_all(inputs, targets):
    # Convert train inputs into one hot representation
    converted_train_inputs = []
    converted_train_targets = []

    for i in range(len(inputs)):

        converted_row = []
        converted1 = convert_one_hot(inputs[i][0])
        converted2 = convert_one_hot(inputs[i][1])
        converted3 = convert_one_hot(inputs[i][2])
        converted_row.append(converted1)
        converted_row.append(converted2)
        converted_row.append(converted3)
        converted_train_inputs.append(converted_row)

        converted_target = convert_one_hot(targets[i])
        converted_train_targets.append(converted_target)

    return converted_train_inputs, converted_train_targets

def load_files():
    train_file = 'data/train_inputs.npy'
    train_file2 = 'data/train_targets.npy'
    test_file = 'data/test_inputs.npy'
    test_file2 = 'data/test_targets.npy'
    validation_file = 'data/valid_inputs.npy'
    validation_file2 = 'data/valid_targets.npy'
    vocab_file = 'data/vocab.npy'

    train_inputs = np.load(train_file)
    train_targets = np.load(train_file2)
    test_inputs = np.load(test_file)
    test_targets = np.load(test_file2)
    valid_inputs = np.load(validation_file)
    valid_targets = np.load(validation_file2)
    vocab = np.load(vocab_file)

    return train_inputs, train_targets, test_inputs, test_targets,valid_inputs, valid_targets, vocab

def calculate_accuracy(network, converted_valid_inputs, valid_targets):
    data_size = len(converted_valid_inputs)
    correct_guess = 0

    for i in range(data_size):
        guess, guess_index = eval.get_prediction(network, converted_valid_inputs[i])
        if guess_index == valid_targets[i]:
            correct_guess += 1

    return correct_guess/data_size

def train(network, converted_train_inputs, converted_train_targets, converted_valid_inputs, valid_targets, learning_rate, batch_size, epochs):

    train_length = len(converted_train_inputs)
    total_batch_number = train_length / batch_size

    epoch_total_losses = []
    epoch_valid_accuracy = []

    # Training loop:
    for e in range(epochs):

        print("\nStart of epoch %d" % (e,))
        start_time = time.time()

        # Shuffle training data:
        converted_train_inputs, converted_train_targets = shuffle(converted_train_inputs, converted_train_targets)

        # Adaptive Learning Rate:
        if e >= 30:
            learning_rate = learning_rate / 10

        batch_total_losses = []

        for b in range(int(total_batch_number)):

            # Get input batch:
            input_batch = converted_train_inputs[b * batch_size:b * batch_size + batch_size]
            target_batch = converted_train_targets[b * batch_size:b * batch_size + batch_size]

            # Forward propagation, training loss:
            average_loss, total_loss, guesses, f_h_batch, s_o_batch, e_batch, o_batch, h_batch = network.forward_propagation_batch(
                batch_size, input_batch, target_batch)

            # Backprop and update:
            dw3, db2, dw2, db1, dw1 = network.backprop(input_batch, target_batch, f_h_batch, s_o_batch, e_batch,
                                                       o_batch, h_batch)
            network.update(dw3, db2, dw2, db1, dw1, learning_rate)

            batch_total_losses.append(total_loss)

            # Log every 100 batches.
            if b % 100 == 0:
                print(
                    "\nTraining loss (for one batch) at step %d: %.4f"
                    % (b, float(total_loss))
                )
                print("Seen so far: %s samples" % ((b + 1) * batch_size))


        batch_total_losses_avg = sum(batch_total_losses) / len(batch_total_losses)
        epoch_total_losses.append(batch_total_losses_avg)

        # Validation set evaluation at the end of epoch:
        valid_accuracy = calculate_accuracy(network, converted_valid_inputs, valid_targets)
        epoch_valid_accuracy.append(valid_accuracy)
        print("\nValidation acc: %.4f" % (float(valid_accuracy),))
        print("Time taken: %.2fs" % (time.time() - start_time))

    plt.plot(epoch_total_losses)
    plt.show()
    plt.plot(epoch_valid_accuracy)
    plt.show()


def main():

    # Load Files:
    train_inputs, train_targets, test_inputs, test_targets, valid_inputs, valid_targets, vocab = load_files()

    # Convert train and validation inputs into one hot representation:
    converted_train_inputs, converted_train_targets = convert_one_hot_all(train_inputs, train_targets)
    converted_valid_inputs, converted_valid_targets = convert_one_hot_all(valid_inputs, valid_targets)
    converted_test_inputs, converted_test_targets = convert_one_hot_all(test_inputs, test_targets)

    # Initialize the Network:
    network = NeuralNetwork()

    # Training Loop:
    train(network, converted_train_inputs,converted_train_targets,
          converted_valid_inputs, valid_targets,
          learning_rate=0.001, batch_size=500, epochs=35)

    # Save the model as pickle
    with open('modelv2.pk', 'wb') as f:
        pickle.dump(network, f)
    f.close()

    # Final validation and test accuracy:
    valid_accuracy = calculate_accuracy(network, converted_valid_inputs, valid_targets)
    train_accuracy = calculate_accuracy(network, converted_train_inputs, train_targets)

    print('final training accuracy is:', train_accuracy)
    print('final validation accuracy is:', valid_accuracy)


if __name__ == '__main__':
    main()