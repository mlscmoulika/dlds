import sys
sys.path.append('C:/Users/MLS Chandra Moulika/Desktop/DLDS/1')
from tqdm import tqdm
import numpy as np
import pandas as pd
import math
import copy
import matplotlib.pyplot as plt

class Classifier:
    def __init__(self, X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, weights, bias, gamma, beta, m, dim_of_image, k, lamba, eta, batch_size, batches, n_layers, batch_norm):
        self.X = X_train
        self.y = y_train
        self.Y = Y_train
        self.X_val = X_val
        self.y_val = y_val
        self.Y_val = Y_val
        self.X_test = X_test
        self.y_test = y_test
        self.Y_test = Y_test

        self.n = self.X.shape[1]
        self.m = m
        self.k = k
        self.dim_of_image = dim_of_image
        self.no_layers = n_layers

        self.lamba = lamba
        self.batches = batches
        self.batch_size = batch_size
        self.eta = eta
        self.alpha = 0.9

        self.weights = weights
        self.bias = bias
        self.beta = beta
        self.gamma = gamma

        self.batch_norm = batch_norm

        # assert dim_of_image == self.X.shape[0]
        # assert k == self.Y.shape[0]

    # CLASSIFICATION FUNCTIONS
    def softmax(self, s):
        return np.exp(s) / np.sum(np.exp(s), axis=0)
    def BatchNormalize(self, s, mu, var):
        return np.multiply(pow(var + np.finfo(float).eps, -1 / 2) , (s - mu))

    def classify(self, x):
        if self.batch_norm:
            s_s = []
            mu_s = []
            var_s = []
            s_hat_s = []
            x_s = []

            x_s.append(x)

            for i in range(self.no_layers - 1):
                s = self.weights[i] @ x + self.bias[i]
                s_s.append(s)
                mu = np.mean(s, axis = 1, keepdims=True)
                var = np.var(s, axis = 1, keepdims=True)
                # assert mu.shape[0] == self.bias[i].shape[0]
                # assert var.shape[0] == mu.shape[0]
                mu_s.append(mu)
                var_s.append(var)
                s = self.BatchNormalize(s, mu, var)

                s_hat = np.multiply(self.gamma[i], s) +self.beta[i]
                s_hat_s.append(s_hat)
                x = np.maximum(0, s_hat)
                x_s.append(x)
            s = self.weights[self.no_layers - 1] @ x + self.bias[self.no_layers - 1]
            p = self.softmax(s.astype(float))
            return p, np.array(x_s, dtype="object"), np.array(s_s, dtype="object"), np.array(s_hat_s, dtype="object"), np.array(mu_s, dtype="object"), np.array(var_s, dtype="object")

        else:
            h = []
            h.append(x)
            for i in range(self.no_layers-1):
                s = self.weights[i] @ x + self.bias[i]
                x = np.maximum(0, s)
                h.append(x)
            s = self.weights[self.no_layers-1] @ x + self.bias[self.no_layers-1]
            # check the dimension of p is it 10,1? yes fine move on:)
            p = self.softmax(s)
            h.append(p)
            # assert p.shape[0] == 10
            return p, np.array(h, dtype="object")

    # MAKE PREDICTION
    def predicted_class(self, x):
        if self.batch_norm:
            p, _, _, _, _, _ = self.classify(x)
        else:
            p, _ = self.classify(x)
        prediction = np.argmax(p, axis=0)
        return prediction

    # COST AND LOSS FUNCTIONS
    def loss(self, x, Y_):
        if self.batch_norm:
            p, _, _, _, _, _ = self.classify(x)
            print("shape of p = "+str(p.shape))
        else:
            p, _ = self.classify(x)
        _log_p = -np.log(np.sum(p * Y_, axis=0))
        loss = np.sum(_log_p)
        return loss / x.shape[1]

    def cost(self, x, Y_):
        cross_loss = self.loss(x, Y_) + self.lamba * (np.sum([np.sum(np.square(w)) for w in self.weights]))
        return cross_loss

    # requires modification
    def ComputeCost(self, x, Y, W, b):
        def classify(self, x):
            if self.batch_norm:
                s_s = []
                mu_s = []
                var_s = []
                s_hat_s = []
                x_s = []

                x_s.append(x)

                for i in range(self.no_layers - 1):
                    s = self.weights[i] @ x + self.bias[i]
                    s_s.append(s)
                    mu = np.mean(s, axis=1, keepdims=True)
                    var = np.var(s, axis=1, keepdims=True)
                    mu_s.append(mu)
                    var_s.append(var)
                    s = self.BatchNormalize(s, mu, var)

                    s_hat = np.multiply(self.gamma[i], s) + self.beta[i]
                    s_hat_s.append(s_hat)
                    x = np.maximum(0, s_hat)
                    x_s.append(x)
                s = self.weights[self.no_layers - 1] @ x + self.bias[self.no_layers - 1]
                p = self.softmax(s.astype(float))
                return p, np.array(x_s, dtype="object"), np.array(s_s, dtype="object"), np.array(s_hat_s,
                                                                                                 dtype="object"), np.array(
                    mu_s, dtype="object"), np.array(var_s, dtype="object")

            else:
                h = []
                h.append(x)
                for i in range(self.no_layers - 1):
                    s = self.weights[i] @ x + self.bias[i]
                    x = np.maximum(0, s)
                    h.append(x)
                s = self.weights[self.no_layers - 1] @ x + self.bias[self.no_layers - 1]
                # check the dimension of p is it 10,1? yes fine move on:)
                p = self.softmax(s)
                h.append(p)
                # assert p.shape[0] == 10
                h = np.array(h, dtype="object")
        for i in range(self.no_layers - 1):
            s = W[i] @ x + b[i]
            x = np.maximum(0, s)
        s = W[self.no_layers - 1] @ x + b[self.no_layers - 1]
        p = self.softmax(s)
        _log_p = -np.log(np.sum(p * Y, axis=0))
        loss = np.sum(_log_p) / x.shape[1]
        cross_loss = loss + self.lamba * (np.sum([np.sum(np.square(w)) for w in W]))
        return cross_loss

    # GRADIENTS CALCULATION AND UPDATION
    def update(self, x, Y):
        if self.batch_norm:
            grad_W, grad_b, grad_gamma, grad_beta = self.ComputeGradients_batch(x, Y)
        grad_W, grad_b = self.ComputeGradients_batch(x, Y)
        for i in range(self.no_layers):
            self.weights[i] -= self.eta * grad_W[i]
            self.b[i] -= self.eta * grad_b[i]
            if self.batch_norm:
                self.gamma[i] -= self.eta * grad_gamma[i]
                self.beta[i] -= self.eta * grad_beta[i]

    def BatchNorm_backpass(self, G, H, mu, var, N):
        sig_1 = np.power(var + np.finfo(float).eps, -0.5)
        sig_2 = np.power(var + np.finfo(float).eps, -1.5)
        G1 = np.multiply(G, sig_1)
        G2 = np.multiply(G, sig_2)
        D = H - mu
        c = np.multiply(G2, D)

        G = G1 - (1/N)*(G1)-(1/N)*np.multiply(D,c)

        return G

    def ComputeGradients_batch(self, x, Y):
        grad_W = copy.deepcopy(self.weights)
        grad_b = copy.deepcopy(self.bias)
        grad_beta = copy.deepcopy(self.beta)
        grad_gamma = copy.deepcopy(self.gamma)
        N = x.shape[1]
        if self.batch_norm:
            P, x_s, S, S_hat, mu_s, var_s = self.classify(x)

            G = -(Y - P)

            grad_W[self.no_layers - 1] = (1 / N) * (G @ x_s[self.no_layers - 1].T) + 2 * self.lamba * self.weights[self.no_layers - 1]
            grad_b[self.no_layers - 1] = np.reshape(1 / N * G @ np.ones(N), (-1, 1))
            # grad_b[self.no_layers - 1] = 1 / N * G @ np.ones(N)

            G = self.weights[self.no_layers - 1].T @ G
            G = np.multiply(G, np.where(x_s[self.no_layers - 1] > 0, 1, 0))

            for l in range(self.no_layers - 2, -1, -1):
                grad_gamma[l] = (1 / N) * np.reshape(np.sum(np.multiply(G, S_hat[l]), axis = 1),(-1,1))
                grad_beta[l] = (1 / N) * np.reshape(np.sum(G, axis = 1),(-1,1))

                G = np.multiply(G, np.sum(self.gamma[l], axis = 0))
                G = self.BatchNorm_backpass(G, S[l], mu_s[l], var_s[l], N)

                grad_W[l] = (1/N) * G @ x_s[l].T + 2 * self.lamba * self.weights[l]
                grad_b[l] = (1/N) * np.sum(G, axis = 1)

                if l-1 != -1:
                    G = self.weights[l].T@G
                    G = np.multiply(G, np.where(x_s[l] > 0, 1, 0))

            return grad_W, grad_b, grad_gamma, grad_beta
        else:
            P, H = self.classify(x)
            # assert (H[0] == x).all()

            I = np.ones(N, dtype=np.float64).reshape(N, 1)

            G = -(Y - P)
            for l in range(self.no_layers-1, -1, -1):
                grad_W[l] = 1/N * (G @ H[l].T) + 2 * self.lamba * self.weights[l]
                grad_b[l] = np.reshape(1/N * G@np.ones(N),(-1, 1))

                G = self.weights[l].T @ G
                G = np.multiply(G, np.where(H[l] > 0, 1, 0))

            return grad_W, grad_b

    # ANALYTICAL AND NUMERICAL GRADIENTS
    def ComputeGradsNum(self, x, Y, h):
        if self.batch_norm:
            P, _, _, _, _, _ = self.classify(x)
        else:
            P, _ = self.classify(x)
        self.W = self.weights
        self.b = self.bias
        grad_W = copy.deepcopy(self.W)
        grad_b = copy.deepcopy(self.b)
        no_of_layers = 2

        c = self.cost(x, Y)
        for j in tqdm(range(no_of_layers)):
            for i in (range(self.b[j].shape[0])):
                b_try = copy.deepcopy(self.b)
                b_try[j][i] += h
                c2 = self.ComputeCost(x, Y, self.W, b_try)
                grad_b[j][i] = (c2 - c) / h

        for j in tqdm(range(self.W.shape[0])):
            for i in (range(self.W[j].shape[0])):
                for k in (range(self.W[j].shape[1])):
                    W_try = copy.deepcopy(self.W)
                    W_try[j][i][k] += h
                    c2 = self.ComputeCost(x, Y, W_try, self.b)
                    grad_W[j][i][k] = (c2 - c) / h

        return grad_W, grad_b

    def ComputeGradsNumSlow(self, X, Y, h):
        self.W = self.weights
        self.b = self.bias
        grad_W = copy.deepcopy(self.W)
        grad_b = copy.deepcopy(self.b)
        no_of_layers = 2

        for j in tqdm(range(no_of_layers)):
            for i in (range(self.b[j].shape[0])):
                b_try = copy.deepcopy(self.b)
                b_try[j][i] = b_try[j][i] - h
                c1 = self.ComputeCost(X, Y, self.W, b_try)

                b_try = copy.deepcopy(self.b)
                b_try[j][i] = b_try[j][i] + h
                c2 = self.ComputeCost(X, Y, self.W, b_try)

                grad_b[j][i] = (c2 - c1) / (2 * h)
        for k in tqdm(range(self.W.shape[0])):
            for i in (range(self.W[k].shape[0])):
                for j in (range(self.W[k].shape[1])):
                    W_try = copy.deepcopy(self.W)
                    W_try[k][i][j] -= h
                    c1 = self.ComputeCost(X, Y, W_try, self.b)

                    W_try = copy.deepcopy(self.W)
                    W_try[k][i][j] += h
                    c2 = self.ComputeCost(X, Y, W_try, self.b)

                    grad_W[k][i][j] = (c2 - c1) / (2 * h)

        return grad_W, grad_b
    def compute_acc(self, x, y):
        predicted_label = np.reshape(self.predicted_class(x), (-1, 1))
        actual_label = np.reshape(y, (-1, 1))
        accuracy = np.sum(predicted_label == actual_label) / len(actual_label)
        accuracy *= 100
        print("overall accuracy = " + str(accuracy) + "%")
        return accuracy
    def train(self, x, Y, y, n_epoches):
        tl = []
        tc = []
        vl = []
        vc = []
        for _ in tqdm(range(n_epoches)):
            P = self.classify(x)
            self.ComputeGradients_batch(x, Y)
            self.update(x, Y)
            tl.append(self.loss(x, Y))
            tc.append(self.cost(x, Y))
            vl.append(self.loss(self.X_val, self.Y_val))
            vc.append(self.cost(self.X_val, self.Y_val))
        return tl, tc, vl, vc
    def train_cyclical(self, eta_min, eta_max, n_s, n_epoches, n_cycles, plot = True):
        tl = []
        tc = []
        vl = []
        vc = []
        t = 0
        l = 0
        for q in range(n_cycles):
            for _ in tqdm(range(n_epoches)):
                for i in range(self.batches):
                    x = self.X[:, (i * self.batch_size):(i + 1) * self.batch_size]
                    Y = self.Y[:, (i * self.batch_size):(i + 1) * self.batch_size]
                    if self.batch_norm:
                        P, H, S_hat, mu, var = self.classify(x)
                    else:
                        P, _ = self.classify(x)
                    self.ComputeGradients_batch(x, Y)
                    self.update(x, Y)
                    if self.batch_norm:
                        if i == 0:
                            mu_av = mu
                            var_av = var
                        else:
                            mu_av = [[self.alpha * x for x in mu_av][y] + [(1 - self.alpha) * x for x in mu][y]
                               for y in range(len(mu))]
                            var_av = [[self.alpha * x for x in var_av][y] + [(1 - self.alpha) * x for x in var][y]
                                     for y in range(len(var))]
                    if plot and t%(n_s//10) == 0:
                        tl_ = self.loss(self.X, self.Y, mu_av, var_av)
                        tl.append(tl_)
                        tc_ = self.cost(self.X, self.Y, mu_av, var_av)
                        tc.append(tc_)
                        vl_ = self.loss(self.X_val, self.Y_val, mu_av, var_av)
                        vl.append(vl_)
                        vc_ = self.cost(self.X_val, self.Y_val, mu_av, var_av)
                        vc.append(vc_)
                    t += 1
                    if (t + 1) % (2 * n_s) == 0:
                        l += 1
                    if (2 * l * n_s <= t and t <= (2 * l + 1) * n_s):
                        val = ((t - (2 * l * n_s)) / n_s) * (eta_max - eta_min)
                        self.eta = eta_min + val
                    elif (((2 * l + 1) * n_s) <= t and t <= 2 * (l + 1) * n_s):
                        self.eta = eta_max - ((t - (2 * l + 1) * n_s) / n_s) * (eta_max - eta_min)
        acc = self.compute_acc(self.X_test, self.y_test, mu_av, var_av)
        if plot:
            plot_cost(tl, vl, len(tl))
            plot_loss(tc, vc, len(tc))
        return tl, tc, vl, vc, acc

def plot_loss(tl, vl, n_epoches):
    epoch = np.arange(n_epoches)
    plt.plot(epoch, tl, "-g", label="train loss")
    plt.plot(epoch, vl, "-r", label="val loss")
    plt.xlabel('epoches')
    plt.ylabel('loss')
    #     plt.title('loss curve '+str(var)+" = "+str(var_val))
    plt.legend(loc="upper left")
    #     plt.savefig(filename+'.png')
    plt.show()


def plot_cost(tc, vc, n_epoches):
    epoch = np.arange(n_epoches)
    plt.plot(epoch, tc, "-g", label="train cost")
    plt.plot(epoch, vc, "-r", label="val cost")
    plt.xlabel('epoches')
    plt.ylabel('cost')
    #     plt.title('cost curve '+str(var)+" = "+str(var_val))
    plt.legend(loc="upper left")
    #     plt.savefig(filename+'.png')
    plt.show()

def LoadBatch(filename):
    final_filename = "C:/Users/MLS Chandra Moulika/Desktop/DLDS/1/Datasets/cifar-10-python.tar/cifar-10-python/cifar-10-batches-py/" + filename
    data = pd.read_pickle(final_filename)
    data['data'] = data['data'].T

    X = np.array(data["data"].T)
    Y = np.eye(10)[data['labels']]
    y = np.array(data['labels'])

    return X, Y, y

def normalise(X, train_mean, train_std, compute_mean = False):
    if compute_mean:
        mean_X = np.mean(X, axis = 0).reshape(-1,1)
        # assert mean_X.shape == (X.shape[1],1)
        std_X = np.std(X, axis = 0).reshape(-1,1)
        # assert std_X.shape == (X.shape[1],1)
        X_normalised = X.T - mean_X
        X_normalised = X_normalised/std_X
        return X_normalised.T, mean_X, std_X
    else:
        X_normalised = X.T - train_mean
        X_normalised = X_normalised/train_std
        return X_normalised.T

def preprocess_ex_1():
    X_train, Y_train, y_train = LoadBatch("data_batch_1")
    X_val, Y_val, y_val = LoadBatch("data_batch_2")
    X_test, Y_test, y_test = LoadBatch("test_batch")

    X_train, train_mean, train_std = normalise(X_train, 0, 0, compute_mean=True)
    X_val = normalise(X_val, train_mean, train_std, False)
    X_test = normalise(X_test, train_mean, train_std, False)

    X_train, Y_train, y_train = np.array(X_train).T, np.array(Y_train).T, np.array(y_train).T
    X_val, Y_val, y_val = np.array(X_val).T, np.array(Y_val).T, np.array(y_val).T
    X_test, Y_test, y_test = np.array(X_test).T, np.array(Y_test).T, np.array(y_test).T

    return X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test

def initialise_w_b(m, dim_of_image, k, n_layers):
    weights = []
    bias = []
    gamma = []
    beta = []

    n = 100000

    w1 = np.random.normal(0, 1 / math.sqrt(dim_of_image), (m[0], dim_of_image))
    b1 = np.random.normal(0, 1 / math.sqrt(dim_of_image), (m[0], 1))
    gamma1 = np.ones((m[0], 1))
    beta1 = np.zeros((m[0], 1))
    weights.append(w1)
    bias.append(b1)
    gamma.append(gamma1)
    beta.append(beta1)
    for i in range(n_layers-1):
        w1 = np.random.normal(0, 1 / math.sqrt(dim_of_image), (m[i+1], m[i]))
        b1 = np.random.normal(0, 1 / math.sqrt(dim_of_image), (m[i+1], 1))
        gamma1 = np.zeros((m[i+1], 1))
        beta1 = np.zeros((m[i+1], 1))
        weights.append(w1)
        bias.append(b1)
        if i+1 != n_layers-1:
            gamma.append(gamma1)
            beta.append(beta1)
    w2 = np.random.normal(0, 1 / math.sqrt(m[-1]), (k, m[-1]))
    b2 = np.random.normal(0, 1 / math.sqrt(m[-1]), (k, 1))
    gamma2 = np.zeros((k, 1))
    beta2 = np.zeros((k, 1))
    # weights.append(w2)
    # bias.append(b2)
    # gamma.append(gamma2)
    # beta.append(beta2)
    weights = np.array(weights, dtype='object')
    bias = np.array(bias, dtype='object')
    beta = np.array(beta, dtype='object')
    gamma = np.array(gamma, dtype='object')

    assert beta.shape[0] == n_layers - 1
    assert beta.shape[0] == n_layers - 1
    assert weights.shape[0] == n_layers
    assert bias.shape[0] == n_layers

    return weights, bias, gamma, beta

def check_grads(classifier, h, n_layers):
    X = classifier.X[:,0:100]
    Y = classifier.Y[:,0:100]
    grad_W_slow, grad_b_slow = classifier.ComputeGradsNumSlow(X, Y, h)
    # grad_W1_slow, grad_W2_slow = grad_W_slow[0][:], grad_W_slow[1][:]
    # grad_b1_slow, grad_b2_slow = np.reshape(grad_b_slow[0][:],(-1,1)),np.reshape(grad_b_slow[1][:],(-1,1))


    grad_W_num, grad_b_num = classifier.ComputeGradsNum(X, Y, h)
    # grad_W1_num, grad_W2_num = grad_W_num[0][:], grad_W_num[1][:]
    # grad_b1_num, grad_b2_num = np.reshape(grad_b_num[0][:], (-1, 1)), np.reshape(grad_b_num[1][:], (-1, 1))
    # print(grad_b1_num)
    grad_W_actual, grad_b_actual = classifier.ComputeGradients_batch(X, Y)
    # grad_W_actual, grad_b_actual, grad_gamma_actual, grad_beta_actual  = classifier.ComputeGradients_batch(X, Y)
    # grad_W1_actual, grad_W2_actual = grad_W_actual[0][:], grad_W_actual[1][:]
    # grad_b1_actual, grad_b2_actual = np.reshape(grad_b_actual[0][:], (-1, 1)), np.reshape(grad_b_actual[1][:], (-1, 1))
    # print(grad_b1_actual)

    print_diff1(grad_b_num, grad_b_slow, grad_b_actual, grad_W_num, grad_W_slow, grad_W_actual, n_layers)
    # print_diff(grad_b1_slow, grad_b1_num, grad_b1_actual, grad_b2_slow, grad_b2_num, grad_b2_actual, grad_W1_slow, grad_W1_num, grad_W1_actual, grad_W2_slow, grad_W2_num, grad_W2_actual)

def print_diff1(grad_b_num, grad_b_analytical, grad_b_actual, grad_W_num, grad_W_analytical, grad_W_actual, n_layers):
    eps = 1e-4
    for i in range(n_layers):
        print("=============================Layer "+str(i)+" =========================================")
        print("Weight1 grad diff - ComputeGradsNumSlow - " + str(
            np.mean(abs(grad_W_num[i] - grad_W_actual[i]) / max(eps, (abs(grad_W_num[i]) + abs(grad_W_actual[i])).all()))))
        print("Weight1 grad diff - ComputeGradsNum - " + str(
            np.mean(abs(grad_W_analytical[i] - grad_W_actual[i]) / max(eps, (abs(grad_W_analytical[i]) + abs(grad_W_actual[i])).all()))))
        print("Bias1 grad diff - ComputeGradsNumSlow - " + str(
            np.mean(abs(np.reshape(grad_b_num[i],(-1,1)) - np.reshape(grad_b_actual[i],(-1,1))) / max(eps, (abs(np.reshape(grad_b_num[i], (-1,1)) )+ abs(np.reshape(grad_b_actual[i],(-1,1)))).all()))))
        print("Bias1 grad diff - ComputeGradsNum - " + str(
            np.mean(abs(np.reshape(grad_b_analytical[i],(-1,1)) - np.reshape(grad_b_actual[i],(-1,1))) / max(eps, (abs(np.reshape(grad_b_analytical[i],(-1,1))) + abs(np.reshape(grad_b_actual[i],(-1,1)))).all()))))


# works only for two layers, I guess it is fine
def print_diff(grad_b1_1, grad_b1_2, grad_b1_3, grad_b2_1, grad_b2_2, grad_b2_3, grad_W1_1, grad_W1_2, grad_W1_3, grad_W2_1, grad_W2_2, grad_W2_3):
    eps = 1e-4
    print("Weight1 grad diff - ComputeGradsNumSlow - " + str(np.mean(abs(grad_W1_1 - grad_W1_3) / max(eps, (abs(grad_W1_1) + abs(grad_W1_3)).all()))))
    print("Weight1 grad diff - ComputeGradsNum - "+str(np.mean(abs(grad_W1_2-grad_W1_3)/max(eps,(abs(grad_W1_2)+abs(grad_W1_3)).all()))))
    print("Bias1 grad diff - ComputeGradsNumSlow - "+str(np.mean(abs(grad_b1_1-grad_b1_3)/max(eps,(abs(grad_b1_1)+abs(grad_b1_3)).all()))))
    print("Bias1 grad diff - ComputeGradsNum - "+str(np.mean(abs(grad_b1_2-grad_b1_3)/max(eps,(abs(grad_b1_2)+abs(grad_b1_3)).all()))))

    print("Weight2 grad diff - ComputeGradsNumSlow - "+str(np.mean(abs(grad_W2_1-grad_W2_3)/max(eps,(abs(grad_W2_1)+abs(grad_W2_3)).all()))))
    print("Weight2 grad diff - ComputeGradsNum - "+str(np.mean(abs(grad_W2_2-grad_W2_3)/max(eps,(abs(grad_W2_2)+abs(grad_W2_3)).all()))))
    print("Bias2 grad diff - ComputeGradsNumSlow - "+str(np.mean(abs(grad_b2_1-grad_b2_3)/max(eps,(abs(grad_b2_1)+abs(grad_b2_3)).all()))))
    print("Bias2 grad diff - ComputeGradsNum - "+str(np.mean(abs(grad_b2_2-grad_b2_3)/max(eps,(abs(grad_b2_2)+abs(grad_b2_3)).all()))))

