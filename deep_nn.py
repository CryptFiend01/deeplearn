import numpy as np
from utils import sigmoid, relu, accuracy

def __init_dnn_parameters(layer_dims):
    params = {}
    L = len(layer_dims)

    for l in range(1, L):
        params['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l - 1]) / np.sqrt(layer_dims[l - 1]) #* 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
    return params

def __activate_forward(A_prev, W, b, activation, keep_prob):
    # A_prev : (n_x, m) W : (n_l, n_x)
    Z = np.dot(W, A_prev) + b
    #Z = W.dot(A_prev) + b
    if activation == 'tanh':
        A = np.tanh(Z)
    elif activation == 'sigmoid':
        A = sigmoid(Z)
    elif activation == 'relu':
        A = relu(Z)

    D = None
    if keep_prob != 1:
        D = np.random.rand(A.shape[0], A.shape[1])
        A = A * D
        A = A / keep_prob
    cache = (A_prev, W, b, Z, D)
    return A, cache

def __dnn_forward(X, params, keep_prob):
    caches = []
    A = X                   # (n_x, m)
    L = len(params) // 2

    for l in range(1, L):
        A_prev = A          # (n_x, m)
        A, cache = __activate_forward(A_prev, params['W' + str(l)], params['b' + str(l)], 'relu', keep_prob)
        caches.append(cache)

    AL, cache = __activate_forward(A, params['W' + str(L)], params['b' + str(L)], 'sigmoid', 1)
    caches.append(cache)
    return AL, caches

def __dnn_compute_cost(AL, Y):
    m = Y.shape[1]
    cost = -(np.dot(Y, np.log(AL).T) + np.dot(1 - Y, np.log(1 - AL).T)) / m
    cost = np.squeeze(cost)
    return cost

def __dnn_compute_cost_regularization(AL, Y, params, lambd):
    m = Y.shape[1]
    L = len(params) // 2
    cost =  __dnn_compute_cost(AL, Y)
    cost_L2 = 0
    for l in range(L):
        cost_L2 += (np.sum(np.square(params['W' + str(l + 1)])))
    cost_L2 = 1.0 / m * lambd / 2 * cost_L2
    return cost + cost_L2

def __linear_backward(dZ, cache, lambd):
    A_prev, W, b, Z, D = cache
    m = A_prev.shape[1]

    dW = 1./ m * np.dot(dZ, A_prev.T)
    if lambd != 0:
        dW = dW + lambd / m * W
    db = 1./ m * dZ.sum(1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)
    return dA_prev, dW, db

def relu_backward(dA, Z):
    dZ = np.array(dA, copy=True)
    dZ[Z <= 0] = 0
    return dZ

def sigmoid_backward(dA, Z):
    s = 1 / (1 + np.exp(-Z))
    dZ = dA * s * (1 - s)
    return dZ

def __activate_backward(dA, cache, activation, lambd, keep_prob):
    A_prev, W, b, Z, D = cache
    if keep_prob != 1:
        dA = dA * D
        dA = dA / keep_prob
    if activation == 'relu':
        dZ = relu_backward(dA, Z)
    elif activation == 'sigmoid':
        dZ = sigmoid_backward(dA, Z)
    dA_prev, dW, db = __linear_backward(dZ, cache, lambd)
    return dA_prev, dW, db

def __dnn_backward(AL, Y, caches, lambd, keep_prob):
    grads = {}
    L = len(caches)
    m = AL.shape[1]
    Y = Y.reshape(AL.shape)

    dAL = -(np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    current_cache = caches[L - 1]
    grads['dA' + str(L)], grads['dW' + str(L)], grads['db' + str(L)] = __activate_backward(dAL, current_cache, 'sigmoid', lambd, 1)

    for l in reversed(range(L - 1)):
        current_cache = caches[l]
        dA = grads['dA' + str(l + 2)]
        grads['dA' + str(l + 1)], grads['dW' + str(l + 1)], grads['db' + str(l + 1)] = __activate_backward(dA, current_cache, 'relu', lambd, keep_prob)
    return grads

def __dnn_update_parameters(params, grads, learning_rate):
    L = len(params) // 2
    for l in range(L):
        params['W' + str(l + 1)] = params['W' + str(l + 1)] - learning_rate * grads['dW' + str(l + 1)]
        params['b' + str(l + 1)] = params['b' + str(l + 1)] - learning_rate * grads['db' + str(l + 1)]
    return params

def __dnn_optimize(X, Y, params, iterations, learning_rate, print_cost, lambd, keep_prob):
    costs = []
    for i in range(iterations):
        AL, caches = __dnn_forward(X, params, keep_prob)

        if lambd == 0:
            cost = __dnn_compute_cost(AL, Y)
        else:
            cost = __dnn_compute_cost_regularization(AL, Y, params, lambd)

        grads = __dnn_backward(AL, Y, caches, lambd, keep_prob)
        params = __dnn_update_parameters(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            print("Cost after iteration %i: %f" % (i, cost))
            costs.append(cost)
    params['costs'] = costs
    return params

def dnn_predict(params, X):
    p = np.zeros((1, X.shape[1]))
    AL, _ = __dnn_forward(X, params, 1)
    #predictions = (AL > 0.5)
    for i in range(AL.shape[1]):
        if AL[0, i] > 0.5:
            p[0, i] = 1
        else:
            p[0, i] = 0
    return p

def dnn_model(X_train, Y_train, X_test, Y_test, layer_dims, iterations=2000, learning_rate = 0.01, print_cost=False, lambd=0, keep_prob=1):
    # X : (n_x, m) Y : (n_y, m)
    dims = []
    dims.append(X_train.shape[0])
    dims += layer_dims
    dims.append(Y_train.shape[0])
    print(dims)
    params = __init_dnn_parameters(dims)
    params = __dnn_optimize(X_train, Y_train, params, iterations, learning_rate, print_cost, lambd, keep_prob)
    
    Y_prediction_test = dnn_predict(params, X_test)
    Y_prediction_train = dnn_predict(params, X_train)
    print("train accuracy: {}%".format(accuracy(Y_train, Y_prediction_train)))
    print("test accuracy: {}%".format(accuracy(Y_test, Y_prediction_test)))
    return params
