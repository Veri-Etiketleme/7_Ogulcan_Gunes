import numpy as np
import random


class LSTM:
    def __init__(self, epoch_size, per_random, rand_mult):
        self.epoch_size = epoch_size
        self.per_random = per_random
        self.rand_mult = rand_mult


    def cross_entropy(self, yhat, y):
        return -1 * np.sum(y * np.log(yhat + 0.001))


    def sigmoid(self, x):
        return 1 / (1 + np.e ** (-1 * x))


    def rand_2D(self, d1, d2):
        return np.random.random_sample((d1, d2)) * self.rand_mult


    def rand_1D(self, d):
        return np.random.random_sample((d)) * self.rand_mult


    def initialize_weights(self, len_input, len_output):
        W = {}
        W['a'] = self.rand_2D(len_output, len_input)
        W['i'] = self.rand_2D(len_output, len_input)
        W['f'] = self.rand_2D(len_output, len_input)
        W['o'] = self.rand_2D(len_output, len_input)

        U = {}
        U['a'] = self.rand_2D(len_output, len_output)
        U['i'] = self.rand_2D(len_output, len_output)
        U['f'] = self.rand_2D(len_output, len_output)
        U['o'] = self.rand_2D(len_output, len_output)

        B = {}
        B['a'] = self.rand_1D(len_output)
        B['i'] = self.rand_1D(len_output)
        B['f'] = self.rand_1D(len_output)
        B['o'] = self.rand_1D(len_output)

        return W, U, B


    def forward_prop(self, X, y_true, W, U, B):
        w_a = W['a']
        w_i = W['i']
        w_f = W['f']
        w_o = W['o']

        u_a = U['a']
        u_i = U['i']
        u_f = U['f']
        u_o = U['o']

        b_a = B['a']
        b_i = B['i']
        b_f = B['f']
        b_o = B['o']

        len_output, len_input = w_a.shape
        num_samples = y_true.shape[0]

        a = np.zeros((num_samples, len_output))
        i = np.zeros((num_samples, len_output))
        f = np.zeros((num_samples, len_output))
        o = np.zeros((num_samples, len_output))
        out = np.zeros((num_samples, len_output))
        state = np.zeros((num_samples, len_output))
        delt = np.zeros((num_samples, len_output))
        cost = np.zeros(num_samples)

        prior_out = out[0, :].T
        prior_state = state[0, :].T

        for t in range(num_samples):
            xt = X[t, :].T
            at = np.tanh(w_a.dot(xt) + u_a.dot(prior_out) + b_a)
            it = self.sigmoid(w_i.dot(xt) + u_i.dot(prior_out) + b_i)
            ft = self.sigmoid(w_f.dot(xt) + u_f.dot(prior_out) + b_f)
            ot = self.sigmoid(w_o.dot(xt) + u_o.dot(prior_out) + b_o)

            state[t, :] = (at * it) + (ft * prior_state)
            out[t, :] = np.tanh(state[t, :] * ot)

            a[t, :] = at
            i[t, :] = it
            f[t, :] = ft
            o[t, :] = ot
            delt[t, :] = y_true[t, :] - out[t, :]

            prior_out = out[t, :]
            prior_state = state[t, :]
            cost[t] = self.cross_entropy(out[t, :], y_true[t, :])

        W_mat = np.concatenate((w_a, w_i, w_f, w_o), axis=0)
        U_mat = np.concatenate((u_a, u_i, u_f, u_o), axis=0)
        B_mat = np.concatenate((b_a, b_i, b_f, b_o), axis=0)
        gates = np.concatenate((a, i, f, o), axis=0)

        return out, state, delt, a, i, f, o, W_mat, U_mat, B_mat, gates, cost


    def forward_validate(self, X_start, dictionary, W, U, B):
        possible_y = np.array(list(dictionary.values()))
        possible_words = np.array(list(dictionary.keys()))

        w_a = W['a']
        w_i = W['i']
        w_f = W['f']
        w_o = W['o']

        u_a = U['a']
        u_i = U['i']
        u_f = U['f']
        u_o = U['o']

        b_a = B['a']
        b_i = B['i']
        b_f = B['f']
        b_o = B['o']

        len_output, len_input = w_a.shape
        num_samples = self.epoch_size

        a = np.zeros((num_samples, len_output))
        i = np.zeros((num_samples, len_output))
        f = np.zeros((num_samples, len_output))
        o = np.zeros((num_samples, len_output))
        out = np.zeros((num_samples, len_output))
        state = np.zeros((num_samples, len_output))
        out_str = ""

        prior_out = out[0, :].T
        prior_state = state[0, :].T
        current_X = X_start

        for t in range(num_samples):
            xt = current_X
            at = np.tanh(w_a.dot(xt) + u_a.dot(prior_out) + b_a)
            it = self.sigmoid(w_i.dot(xt) + u_i.dot(prior_out) + b_i)
            ft = self.sigmoid(w_f.dot(xt) + u_f.dot(prior_out) + b_f)
            ot = self.sigmoid(w_o.dot(xt) + u_o.dot(prior_out) + b_o)

            state[t, :] = (at * it) + (ft * prior_state)
            out[t, :] = np.tanh(state[t, :] * ot)

            prior_out = out[t, :]
            prior_state = state[t, :]

            cost = np.zeros(possible_y.shape[0])
            for m in range(possible_y.shape[0]):
                cost[m] = self.cross_entropy(prior_out, possible_y[m, :])

            min_indexes = np.argsort(cost)[:self.per_random]
            chosen_index = np.random.choice(min_indexes)
            out_str += possible_words[chosen_index] + " "
            current_X = possible_y[chosen_index]

        return out_str


    def back_prop(self, X, out, state, delt, a, i, f, o, W, U, B, gates):
        deltW = np.zeros(W.shape)
        deltU = np.zeros(U.shape)
        deltB = np.zeros(B.shape)

        ndx = out.shape[0] - 1
        d_out = delt[ndx, :]
        d_state = d_out * o[ndx, :] * (1 - (np.tanh(state[ndx, :])**2))
        d_a = d_state * i[ndx, :] * (1 - (a[ndx, :])**2)
        d_i = d_state * a[ndx, :] * i[ndx, :] * (1 - i[ndx, :])
        d_f = d_state * state[ndx - 1, :] * f[ndx, :] * (1 - f[ndx, :])
        d_o = d_out * np.tanh(state[ndx, :]) * o[ndx, :] * (1 - o[ndx, :])
        d_gates = np.concatenate((d_a, d_i, d_f, d_o), axis=0)
        d_x = W.T @ d_gates
        gradient_loss = U.T @ d_gates

        deltW += np.outer(d_gates, X[ndx, :])
        d_gates_tplusone = d_gates
        curdelt_state = d_state

        for t in range(out.shape[0] - 2, 0, -1):
            d_out = delt[t, :] + gradient_loss
            d_state = d_out * o[t, :] * (1 - (np.tanh(state[t, :])**2)) + curdelt_state * f[t + 1, :]
            d_a = d_state * i[t, :] * (1 - (a[t, :])**2)
            d_i = d_state * a[t, :] * i[t, :] * (1 - i[t, :])
            d_f = d_state * state[t - 1, :] * f[t, :] * (1 - f[t, :])
            d_o = d_out * np.tanh(state[t, :]) * o[t, :] * (1 - o[t, :])
            d_gates = np.concatenate((d_a, d_i, d_f, d_o), axis=0)
            d_x = W.T @ d_gates
            gradient_loss = U.T @ d_gates

            deltW += np.outer(d_gates, X[t, :])
            deltU += np.outer(d_gates_tplusone, out[t, :])
            deltB += d_gates_tplusone

            d_gates_tplusone = d_gates
            curdelt_state = d_state

        ndx = 0
        d_out = delt[ndx, :] + gradient_loss
        d_state = d_out * o[ndx, :] * (1 - (np.tanh(state[ndx, :])**2)) + curdelt_state * f[ndx + 1, :]
        d_a = d_state * i[ndx, :] * (1 - (a[ndx, :])**2)
        d_i = d_state * a[ndx, :] * i[ndx, :] * (1 - i[ndx, :])
        d_f = d_state * 0
        d_o = d_out * np.tanh(state[ndx, :]) * o[ndx, :] * (1 - o[ndx, :])
        d_gates = np.concatenate((d_a, d_i, d_f, d_o), axis=0)
        d_x = W.T @ d_gates

        deltW += np.outer(d_gates, X[ndx, :])
        deltU += np.outer(d_gates_tplusone, out[ndx, :])
        deltB += d_gates_tplusone

        W += deltW
        U += deltU
        B += deltB

        w_split = int(W.shape[0] / 4)
        u_split = int(U.shape[0] / 4)
        b_split = int(B.shape[0] / 4)

        Wnu = {}
        Unu = {}
        Bnu = {}

        Wnu['a'] = W[0:w_split, :]
        Wnu['i'] = W[(w_split):(2 * w_split), :]
        Wnu['f'] = W[(2 * w_split):(3 * w_split), :]
        Wnu['o'] = W[(3 * w_split):(4 * w_split), :]

        Unu['a'] = U[0:u_split, :]
        Unu['i'] = U[(u_split):(2 * u_split), :]
        Unu['f'] = U[(2 * u_split):(3 * u_split), :]
        Unu['o'] = U[(3 * u_split):(4 * u_split), :]

        Bnu['a'] = B[0:b_split]
        Bnu['i'] = B[(b_split):(2 * b_split)]
        Bnu['f'] = B[(2 * b_split):(3 * b_split)]
        Bnu['o'] = B[(3 * b_split):(4 * b_split)]

        return Wnu, Unu, Bnu

    def calibrate(self, X, y_true):
        num_samples, len_input = X.shape
        len_output = y_true.shape[1]
        assert(num_samples == y_true.shape[0])
        num_epoch = int(num_samples / self.epoch_size)

        W, U, B = self.initialize_weights(len_input, len_output)
        cost = np.zeros(num_samples)

        print(f"\n\nAbout to iterate through {num_epoch} epochs of size {self.epoch_size}. Hold onto your hats\n\n")

        for n in range(num_epoch):
            indexes = range(self.epoch_size * n, (self.epoch_size * (n + 1) - 1))
            ndx_plus_one = [ndx + 1 for ndx in indexes]
            X_epoch = X[indexes, :]
            y_epoch = y_true[ndx_plus_one, :]

            out, state, delt, a, i, f, o, W_mat, U_mat, B_mat, gates, cost_split = self.forward_prop(X_epoch, y_epoch, W, U, B)
            cost[indexes] = cost_split
            W, U, B = self.back_prop(X_epoch, out, state, delt, a, i, f, o, W_mat, U_mat, B_mat, gates)
            print(n)

        result = {}
        result['cost'] = cost
        result['W'] = W
        result['U'] = U
        result['B'] = B
        return result


    def next_word_predict(self, original_txt, word_vecs):
        X = np.zeros((len(original_txt) - 1, len(word_vecs)))
        Y = np.zeros((len(original_txt) - 1, len(word_vecs)))

        for m in range(len(original_txt) - 1):
            X[m, :] = word_vecs[original_txt[m]]
            Y[m, :] = word_vecs[original_txt[m + 1]]

        result = self.calibrate(X, Y)
        return result


    def validate_strings(self, word_vecs, W, U, B, num_strings):
        start_indexes = np.array(range(len(word_vecs)))
        np.random.shuffle(start_indexes)

        keys = list(word_vecs.keys())
        out_strings = []

        for c in range(num_strings):
            word = random.choice(keys)
            x_current = word_vecs[word]
            out_strings.append(word + " " + self.forward_validate(x_current, word_vecs, W, U, B))

        return out_strings
