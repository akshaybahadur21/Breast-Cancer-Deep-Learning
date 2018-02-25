"""
Deep Network for breast cancer classification
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def main():
    train_x = pd.read_csv("cancer_data.csv")
    train_x = np.array(train_x)
    train_y = pd.read_csv("cancer_data_y.csv")
    train_y = np.array(train_y)
    #dims = [30, 30, 20, 11, 1]
    dims = [30, 50, 20, 11, 1]
    d = model(train_x.T, train_y.T, dims, print_cost=True)


def sigmoid(z):
    """

    :param z:
    :return:
    """
    s = 1 / (1 + np.exp(-z))
    cache = z
    return s, cache


def relu(z):
    """

    :param z:
    :return:
    """
    s = np.maximum(0, z)
    cache = z
    return s, cache


def sigmoid_backward(dA, cache):
    """

    :param dA:
    :param activation_cache:
    :return:
    """
    Z = cache
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ


def relu_backward(dA, cache):
    """

    :param dA:
    :param activation_cache:
    :return:
    """
    Z = cache
    dZ = np.array(dA, copy=True)  # just converting dz to a correct object.
    dZ[Z <= 0] = 0
    return dZ


def initialize_parameters_deep(dims):
    """

    :param dims:
    :return:
    """

    np.random.seed(3)
    params = {}
    L = len(dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(dims[l], dims[l - 1]) * 0.01
        params['b' + str(l)] = np.zeros((dims[l], 1))
    return params


def linear_forward(A, W, b):
    """

    :param A:
    :param W:
    :param b:
    :return:
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """

    :param A_prev:
    :param W:
    :param b:
    :param activation:
    :return:
    """
    if activation == "sigmoid":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    elif activation == "relu":
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    cache = (linear_cache, activation_cache)

    return A, cache


def L_model_forward(X, params):
    """

    :param X:
    :param params:
    :return:
    """

    caches = []
    A = X
    L = len(params) // 2  # number of layers in the neural network

    # Implement [LINEAR -> RELU]*(L-1). Add "cache" to the "caches" list.
    for l in range(1, L):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             params["W" + str(l)],
                                             params["b" + str(l)],
                                             activation='relu')
        caches.append(cache)

    A_last, cache = linear_activation_forward(A,
                                              params["W" + str(L)],
                                              params["b" + str(L)],
                                              activation='sigmoid')
    caches.append(cache)
    return A_last, caches


def compute_cost(A_last, Y):
    """

    :param A_last:
    :param Y:
    :return:
    """

    m = Y.shape[1]
    cost = (-1. / m) * np.sum(np.multiply(Y, np.log(A_last)) + np.multiply((1 - Y), np.log(1 - A_last)))
    cost = np.squeeze(cost)  # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
    return cost


def linear_backward(dZ, cache):
    """

    :param dZ:
    :param cache:
    :return:
    """

    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1. / m) * np.dot(dZ, cache[0].T)
    db = (1. / m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(cache[1].T, dZ)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """

    :param dA:
    :param cache:
    :param activation:
    :return:
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_backward(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def L_model_backward(A_last, Y, caches):
    """

    :param A_last:
    :param Y:
    :param caches:
    :return:
    """

    grads = {}
    L = len(caches)  # the number of layers
    m = A_last.shape[1]
    Y = Y.reshape(A_last.shape)  # after this line, Y is the same shape as A_last

    dA_last = - (np.divide(Y, A_last) - np.divide(1 - Y, 1 - A_last))
    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dA_last,
                                                                                                  current_cache,
                                                                                                  activation="sigmoid")

    for l in reversed(range(L - 1)):
        current_cache = caches[l]

        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache,
                                                                    activation="relu")
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_params(params, grads, alpha):
    """

    :param params:
    :param grads:
    :param alpha:
    :return:
    """

    L = len(params) // 2  # number of layers in the neural network

    for l in range(L):
        params["W" + str(l + 1)] = params["W" + str(l + 1)] - alpha * grads["dW" + str(l + 1)]
        params["b" + str(l + 1)] = params["b" + str(l + 1)] - alpha * grads["db" + str(l + 1)]

    return params


def model(X, Y, layers_dims, alpha=0.009, num_iterations=2000, print_cost=True):  # lr was 0.009
    """
    Implements a L-layer neural network: [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID.

    Arguments:
    X -- data, numpy array of shape (number of examples, num_px * num_px * 3)
    Y -- true "label" vector (containing 0 if cat, 1 if non-cat), of shape (1, number of examples)
    layers_dims -- list containing the input size and each layer size, of length (number of layers + 1).
    alpha -- learning rate of the gradient descent update rule
    num_iterations -- number of iterations of the optimization loop
    print_cost -- if True, it prints the cost every 100 steps

    Returns:
    params -- params learnt by the model. They can then be used to predict.
    """

    np.random.seed(1)
    costs = []  # keep track of cost

    params = initialize_parameters_deep(layers_dims)

    for i in range(0, num_iterations):

        A_last, caches = L_model_forward(X, params)
        cost = compute_cost(A_last, Y)
        grads = L_model_backward(A_last, Y, caches)
        if (i > 800):
            alpha1 = (150 / (1.5*i)) * alpha
            params = update_params(params, grads, alpha1)
        else:
            params = update_params(params, grads, alpha)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
        if print_cost and i % 100 == 0:
            costs.append(cost)
    predictions = predict(params, X)
    print('\nAccuracy on training set: %d' % float(
        (np.dot(Y, predictions.T) + np.dot(1 - Y, 1 - predictions.T)) / float(Y.size) * 100) + '%')
    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    falsePositive = 0
    predictions = predictions.astype(int)
    predList = predictions.tolist()
    tlist = Y.tolist()

    array_length = len(predList[0])
    for i in range(array_length):
        if predList[0][i] == 1 and tlist[0][i] == 1:
            truePositive += 1
        elif predList[0][i] == 0 and tlist[0][i] == 0:
            trueNegative += 1
        elif predList[0][i] == 0 and tlist[0][i] == 1:
            falseNegative += 1
        elif predList[0][i] == 1 and tlist[0][i] == 0:
            falsePositive += 1
        else:
            print(predList[0][i])
            print(tlist[0][i])
            print("WTF")
    tpr=truePositive/(truePositive+falseNegative)
    fpr=falsePositive/(falsePositive+trueNegative)
    print("\nOn Train set:\nTrue Positive:  ", truePositive)
    print("True Negative:  ", trueNegative)
    print("False Negative:  ", falseNegative)
    print("False Positive:  ", falsePositive)
    print("True Positive Rate / Recall:  ", tpr)
    print("False Positive Rate / Fallout:  ", fpr)

    X_test = pd.read_csv("test_cancer_data.csv")
    X_test = np.array(X_test)
    X_test = X_test.T
    Y_test = pd.read_csv("test_cancer_data_y.csv")
    Y_test = np.array(Y_test)
    Y_test = Y_test.T

    predictions = predict(params, X_test)
    print('\nAccuracy on test set: %d' % float(
        (np.dot(Y_test, predictions.T) + np.dot(1 - Y_test, 1 - predictions.T)) / float(Y_test.size) * 100) + '%')
    truePositive = 0
    trueNegative = 0
    falseNegative = 0
    falsePositive = 0
    predList = predictions.tolist()
    tlist = Y_test.tolist()

    array_length = len(predList[0])
    for i in range(array_length):
        if predList[0][i] == 1 and tlist[0][i] == 1:
            truePositive += 1
        elif predList[0][i] == 0 and tlist[0][i] == 0:
            trueNegative += 1
        elif predList[0][i] == 0 and tlist[0][i] == 1:
            falseNegative += 1
        elif predList[0][i] == 1 and tlist[0][i] == 0:
            falsePositive += 1
        else:
            print(predList[0][i])
            print(tlist[0][i])
            print("WTF")

    tpr=truePositive/(truePositive+falseNegative)
    fpr=falsePositive/(falsePositive+trueNegative)
    print("\nOn Test set:\nTrue Positive:  ", truePositive)
    print("True Negative:  ", trueNegative)
    print("False Negative:  ", falseNegative)
    print("False Positive:  ", falsePositive)
    print("True Positive Rate / Recall:  ", tpr)
    print("False Positive Rate / Fallout:  ", fpr)

    plt.plot(np.squeeze(costs))
    plt.ylabel('cost')
    plt.xlabel('iterations (per tens)')
    plt.title("Learning rate =" + str(alpha))
    plt.show()

    return params


def predict(parameters, X):
    A_last, cache = L_model_forward(X, parameters)
    predictions = np.round(A_last)

    return predictions


main()
