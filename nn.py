import numpy as np
from utils import sigmoid, relu, accuracy, makeZ

def __init_nn_parameters(n_x, n_h, n_y):
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h) * 0.01
    b2 = np.zeros((n_y, 1))

    params = {"W1" : W1, "b1" : b1, "W2" : W2, "b2" : b2}
    return params

def __forward_propagation(X, params):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    Z1 = np.dot(W1, X) + b1  # (n_h, m)
    A1 = np.tanh(Z1)         # (n_h, m)
    Z2 = np.dot(W2, A1) + b2  # (n_y, m)
    A2 = sigmoid(Z2)        # (n_y, m)

    cache = {"Z1" : Z1, "A1" : A1, "Z2" : Z2, "A2" : A2}
    return A2, cache

def __compute_cost(A2, Y, params):
    # A2.shape : (n_y, m) Y.shape : (n_y, m)
    m = Y.shape[1]
    cost = (1./m) * (-np.dot(Y, np.log(A2).T) - np.dot(1 - Y, np.log(1 - A2).T))
    cost = np.squeeze(cost)
    return cost

def __backward_propagation(params, cache, X, Y):
    # X : (n_x, m), Y : (n_y, m)
    W1 = params['W1']   # (n_h, n_x)
    W2 = params['W2']   # (n_y, n_h)

    A1 = cache['A1']    # (n_h, m)
    A2 = cache['A2']    # (n_y, m)

    m = X.shape[1]

    dZ2 = A2 - Y                                    # (n_y, m)
    dW2 = np.dot(dZ2, A1.T) / m                     # (n_y, n_h)
    db2 = np.sum(dZ2, axis=1, keepdims=True) / m    # (n_y, m)
    dZ1 = np.dot(W2.T, dZ2) * (1 - np.power(A1, 2)) # ()
    dW1 = np.dot(dZ1, X.T) / m
    db1 = np.sum(dZ1, axis=1, keepdims=True) / m

    grads = {'dW1': dW1, 'db1': db1, "dW2": dW2, "db2": db2}
    return grads

def __update_parameters(params, grads, learning_rate):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    dW1 = grads['dW1']
    db1 = grads['db1']
    dW2 = grads['dW2']
    db2 = grads['db2']

    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

    params = {"W1": W1, "b1": b1, "W2": W2, "b2": b2}
    return params

def __nn_optimize(params, X, Y, iterations, learning_rate, print_cost):
    costs = []
    for i in range(iterations):
        A2, cache = __forward_propagation(X, params)

        cost = __compute_cost(A2, Y, params)

        grads = __backward_propagation(params, cache, X, Y)

        params = __update_parameters(params, grads, learning_rate)

        if print_cost and i % 100 == 0:
            costs.append(cost)
            print("Cost after iteration %i : %f" % (i, cost))
    params['costs'] = costs
    return params

def nn_predict(params, X):
    A2, _ = __forward_propagation(X, params)
    prediction = (A2 > 0.5)
    return prediction

def nn_model(X_train, Y_train, X_test, Y_test, n_h, iterations=2000, learning_rate = 0.01, print_cost=False):
    # X : (n_x, m), Y : (n_y, m)
    X = X_train
    Y = Y_train
    np.random.seed(10)
    n_x = X.shape[0]
    n_y = Y.shape[0]
    params = __init_nn_parameters(n_x, n_h, n_y)
    params = __nn_optimize(params, X, Y, iterations, learning_rate, print_cost)

    Y_prediction_test = nn_predict(params, X_test)
    Y_prediction_train = nn_predict(params, X_train)
    print("train accuracy: {}%".format(accuracy(Y_train, Y_prediction_train)))
    print("test accuracy: {}%".format(accuracy(Y_test, Y_prediction_test)))
    return params

if __name__ == "__main__":
    p = __init_nn_parameters(3, 4, 1)
    print(p['W1'].shape)
    print(p['b1'].shape)
    print(p['W2'].shape)
    print(p['b2'].shape)