import csv
import numpy as np


class SkipGram:
    def __init__(self, txt_file, options):
        self.learning_rate = options['learning_rate']
        self.projection_size = options['projection_size']
        self.txt_file = txt_file
        self.rand_mult = options['rand_mult']
        self.nearest_words = options['nearest_words']


    def load_and_clean(self, txt_file):
        rdr = csv.reader(open(txt_file, "r"), delimiter=' ')
        clean_string = []

        for row in rdr:
            for i in range(len(row)):
                wd = row[i].lower()
                b = 0
                illegal_chars = "[]()'&-*+;.? "
                if any(char == ill_char or char.isdigit() for char in wd for ill_char in illegal_chars):
                    b = 1
                elif wd.find(',') == (len(wd) - 1):
                    clean_string.append(wd[:(len(wd) - 1)])
                elif wd.find(',') != -1:
                    b = 1
                else:
                    clean_string.append(wd)
                
        return clean_string

    def create_dictionary(self, original_text):
        unique_words = original_text.copy()

        ndx = 0
        unique_words.sort()
        while True:
            if unique_words[ndx] == unique_words[ndx + 1]:
                unique_words.pop(ndx + 1)
            else:
                ndx += 1
            if len(unique_words) <= (ndx + 1):
                break
        return unique_words

    def gen_2D_rand(self, d1, d2):
        return np.random.random_sample((d1, d2)) * self.rand_mult

    def initialize_weights(self):
        size_dict = len(self.dictionary)
        size_proj = self.projection_size

        W1 = self.gen_2D_rand(size_dict, size_proj)
        W2 = self.gen_2D_rand(size_proj, size_dict)
        return W1, W2


    def softmax(self, z):
        return np.exp(z) / np.sum(np.exp(z))


    def cross_entropy(self, y_pred, y):
        n, m = y.shape
        sum = 0
        yhat = np.zeros((y_pred.shape[0], 4))
        yhat[:, 0] = y_pred
        yhat[:, 1] = y_pred
        yhat[:, 2] = y_pred
        yhat[:, 3] = y_pred
        for j in range(m):
            for i in range(n):
                sum += y[i,j] * np.log(yhat[i,j] + 0.001)
        return (-1 / m) * sum


    def hot_encode(self, word):
        vect = np.zeros(len(self.dictionary))
        vect[self.dictionary.index(word)] = 1
        return vect


    def forward_prop(self, x_hot, W1, W2):
        projection = x_hot.dot(W1)
        z = projection.dot(W2).T
        yhat = self.softmax(z)
        return yhat, projection


    def back_prop(self, projection, yhat, Y, W2):
        dL_dZ = yhat - Y
        dL_dW2 = np.outer(dL_dZ, projection)
        dL_dW1 = W2.dot(dL_dZ)
        return dL_dW1, dL_dW2


    def calibrate(self):
        self.original_text = self.load_and_clean(self.txt_file)
        self.dictionary = self.create_dictionary(self.original_text)
        W1, W2 = self.initialize_weights()

        x_train = self.original_text
        cost = np.zeros(len(x_train) - 4)
        ndy = 0

        for n in range(self.nearest_words, len(x_train) - self.nearest_words):
            curr_word = x_train[n]
            x_hot = self.hot_encode(curr_word)

            ndx = 0
            deltaW1 = np.zeros(W1.shape)
            deltaW2 = np.zeros(W2.shape)
            y_true = np.empty((len(self.dictionary), 4))

            yhat, proj = self.forward_prop(x_hot, W1, W2)
            
            iterate_nearest = list(range(-1 * self.nearest_words, 0))
            for n in range(1, self.nearest_words + 1):
                iterate_nearest.append(n)

            for m in iterate_nearest:
                ytrue_hot = self.hot_encode(x_train[n+m])
                y_true[:, ndx] = ytrue_hot
                dL_dW1, dL_dW2 = self.back_prop(proj, yhat, ytrue_hot, W2)
                deltaW1 += dL_dW1
                deltaW2 += dL_dW2.T

                ndx += 1

            W1 -= self.learning_rate * deltaW1 / 4
            W2 -= self.learning_rate * deltaW2 / 4

            cost[ndy] = self.cross_entropy(yhat, y_true)
            print(n)
            ndy += 1
        out = {}
        out['W1'] = W1
        out['W2'] = W2
        out['cost'] = cost

        return out


    def train_to_vectors(self):
        input = self.calibrate()
        W1 = input['W1']
        W2 = input['W2']

        word_vecs = {}

        for word in self.dictionary:
            x_hot = self.hot_encode(word)
            y_out = self.softmax(x_hot.dot(W1).dot(W2).T)
            word_vecs[word] = y_out

        return word_vecs, input['cost']
