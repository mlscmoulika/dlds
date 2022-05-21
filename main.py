import data_utils
import numpy as np

def exercise_1(m, dim_of_image, k, lamba, eta, batch_size, n_layers, batch_norm):
    X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test = data_utils.preprocess_ex_1()
    n = X_train.shape[1]
    batches = int(n / batch_size)
    weights, bias, gamma, beta = data_utils.initialise_w_b(m, dim_of_image, k, n_layers)
    classifier = data_utils.Classifier(X_train, Y_train, y_train, X_val, Y_val, y_val, X_test, Y_test, y_test, weights, bias, gamma, beta, m, dim_of_image, k, lamba, eta, batch_size, batches, n_layers, batch_norm)
    return classifier
def exercise_3(m, dim_of_image, k, lamba, eta, batch_size, n_layers, batch_norm):
    classifier = exercise_1(m, dim_of_image, k, lamba, eta, batch_size, n_layers, batch_norm)
    data_utils.check_grads(classifier, 1e-8, n_layers)
    return classifier

def exercise_2():
    pass

def exercise_4():
    pass

def main():
    # hyperparameters
    dim_of_image = 3072
    k = 10
    lamba = 0.1
    eta = 0.001
    batch_size = 100
    # last layer is output layer always 10 so only 3 inputs ie hidden layers in m
    n_layers = 3
    m = [50, 50, 10]
    # Exercise 1
    batch_norm = False
    classifier = exercise_1(m, dim_of_image, k, lamba, eta, batch_size, n_layers, batch_norm)
    classifier.ComputeGradients_batch(classifier.X[:,0:10], classifier.y[0:10])
    data_utils.check_grads(classifier, 1e-8, n_layers)
    # Exercise 2
    # question 1
    # batches = 10000/batch_size
    # n_s = 5 * 45,000 / batches

    # trial for different layers
    # Exercise 3
    # batch_norm = True
    # classifier = exercise_3(m, dim_of_image, k, lamba, eta, batch_size, n_layers, batch_norm)
    # classifier.classify(classifier.X[:, 0:11])
    # classifier.ComputeGradients_batch(classifier.X[:, 0:100], classifier.y[0:100])

    # Exercise 4
    exercise_4()


if __name__ == '__main__':
    main()

