import numpy as np
import pickle
from network import NeuralNetwork

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

    return train_inputs, train_targets, test_inputs, test_targets, valid_inputs, valid_targets, vocab

def softmax(x):
    # Numerically stable softmax based on
    # http://timvieira.github.io/blog/post/2014/02/11/exp-normalize-trick/
    b = x.max()
    y = np.exp(x - b)
    return y / y.sum()

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def get_prediction(model, row):
    # row = [1x250, 1x250, 1x250] one input -> x1,x2,x3
    # y is real y for that row

    # 1. Embedding Layer
    e1 = row[0] @ model.network[0]  # OK -> 1x16
    e2 = row[1] @ model.network[0]  # OK -> 1x16
    e3 = row[2] @ model.network[0]  # OK -> 1x16

    e = np.concatenate([e1, e2, e3])
    e = np.reshape(e, (-1, 48))  # OK -> 1x48

    # 2. Hidden Layer
    h = e @ model.network[1]
    h = np.reshape(h, (-1, 128))  # OK -> 1x128
    h = h + model.network[2]  # OK -> 1x128

    # 3. Sigmoid Activation
    f_h = sigmoid(h)  # OK(?) -> 1x128

    # 4. Output layer
    o = f_h @ model.network[3]  # OK -> 1x250
    o = o + model.network[4]  # OK -> 1x250

    # 5. Softmax
    s_o = softmax(o)  # OK(?) -> 1x250

    # Optional: Get one hot encoding of prediction
    guess = np.zeros_like(s_o)
    max = s_o[0][0]
    max_index = 0

    for i in range(250):
        if s_o[0][i] > max:
            max = s_o[0][i]
            max_index = i
    guess[0][max_index] = 1

    return guess, max_index

def convert_one_hot(word_index):
    one_hot_representation = np.zeros(250)
    one_hot_representation[word_index] = 1
    return one_hot_representation

def convert_one_hot_all_test(test_inputs, test_targets):
    # Convert train inputs into one hot representation
    converted_test_inputs = []
    converted_test_targets = []

    for i in range(len(test_inputs)):
        converted_row = []
        converted1 = convert_one_hot(test_inputs[i][0])
        converted2 = convert_one_hot(test_inputs[i][1])
        converted3 = convert_one_hot(test_inputs[i][2])
        converted_row.append(converted1)
        converted_row.append(converted2)
        converted_row.append(converted3)
        converted_test_inputs.append(converted_row)

        converted_target = convert_one_hot(test_targets[i])
        converted_test_targets.append(converted_target)

    return converted_test_inputs, converted_test_targets

def convert_one_hot_to_index(one_hot_vector):
    index = 0

    for i in range(len(one_hot_vector)):
        if one_hot_vector[i] != 0.0:
            index = i
    return index


def convert_word_to_index(words, word):
    index = 0
    for i in range(len(words)):
        if words[i] == word:
            index = i

    return index


def guess_next_word(model, words, word1, word2, word3):
    word_index_1 = convert_word_to_index(words, word1)
    word_index_2 = convert_word_to_index(words, word2)
    word_index_3 = convert_word_to_index(words, word3)

    test_row = []

    word_1 = convert_one_hot(word_index_1)
    word_2 = convert_one_hot(word_index_2)
    word_3 = convert_one_hot(word_index_3)

    test_row.append(word_1)
    test_row.append(word_2)
    test_row.append(word_3)

    guess, guess_index = get_prediction(model, test_row)
    guessed_word = words[guess_index]

    print(word1, ' ', word2, ' ', word3, ' ', guessed_word)


def get_test_accuracy(model, converted_test_inputs, test_targets):
    data_size = len(converted_test_inputs)
    correct_guess = 0

    for i in range(data_size):
        guess, guess_index = get_prediction(model, converted_test_inputs[i])
        if guess_index == test_targets[i]:
            correct_guess += 1

    print('Test accuracy is:', correct_guess / data_size)


def main():
    # Load Files
    train_inputs, train_targets, test_inputs, test_targets, valid_inputs, valid_targets, vocab = load_files()

    # Convert test inputs into one hot representation
    converted_test_inputs, converted_test_targets = convert_one_hot_all_test(test_inputs, test_targets)

    file = open("model.pk", 'rb')
    my_model = pickle.load(file)
    file.close()

    words = np.load('data/vocab.npy')

    # Guessing test:

    guess_next_word(my_model, words, 'city', 'of', 'new')
    guess_next_word(my_model, words, 'life', 'in', 'the')
    guess_next_word(my_model, words, 'he', 'is', 'the')
    guess_next_word(my_model, words, 'world', 'is', 'a')
    guess_next_word(my_model, words, 'where', 'is', 'the')
    guess_next_word(my_model, words, 'how', 'are', 'the')

    # Get the test accuracy:
    get_test_accuracy(my_model, converted_test_inputs, test_targets)


if __name__ == '__main__':
    main()