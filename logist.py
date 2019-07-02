import numpy as np
from utils import sigmoid, accuracy, makeZ

def __init_logist_parameters(dim):
    w = np.zeros((dim, 1))
    b = 0
    assert(w.shape == (dim, 1))
    assert(isinstance(b, float) or isinstance(b, int))
    return w, b

def __propagate(w, b, X, Y):
    m = X.shape[1]
    Z = makeZ(w, X, b)
    A = sigmoid(Z)
    cost = -np.sum(Y * np.log(A) + (1 - Y) * np.log(1 - A)) / m
    dw = np.dot(X, (A - Y).T) / m
    db = np.sum(A - Y) / m

    assert(dw.shape == w.shape)
    assert(db.dtype == float)
    cost = np.squeeze(cost)
    assert(cost.shape == ())
    grads = {"dw": dw, "db": db}
    #print(grads)
    return grads, cost

def __optimize(w, b, X, Y, iterations, learning_rate, print_cost):
    costs = []
    for i in range(iterations):
        grads, cost = __propagate(w, b, X, Y)

        dw = grads['dw']
        db = grads['db']

        w = w - learning_rate * dw
        b = b - learning_rate * db

        if i % 100 == 0:
            costs.append(cost)
            if print_cost:
                print("Cost after iteration %i: %f" % (i, cost))
    params = {"w": w, "b": b}
    grads = {"dw": dw, "db": db}
    return params, grads, costs

def logist_predict(w, b, X):
    Y = np.zeros((1, X.shape[1]))
    A = sigmoid(makeZ(w, X, b))
    for i in range(A.shape[1]):
        if A[0, i] < 0.5:
            Y[0, i] = 0
        else:
            Y[0, i] = 1
    return Y

def logist_model(X_train, Y_train, X_test, Y_test, iterations = 2000, learning_rate = 0.5, print_cost = False):
    # X : (n_x, m), Y : (1, m), w : (n_x, 1), b : numeric
    w, b = __init_logist_parameters(X_train.shape[0])
    parameters, grads, costs = __optimize(w, b, X_train, Y_train, iterations, learning_rate, print_cost)
    w = parameters['w']
    b = parameters['b']
    Y_prediction_test = logist_predict(w, b, X_test)
    Y_prediction_train = logist_predict(w, b, X_train)
    print("train accuracy: {}%".format(accuracy(Y_train, Y_prediction_train)))
    print("test accuracy: {}%".format(accuracy(Y_test, Y_prediction_test)))

    d = { "costs": costs, "w": w, "b": b }
    return d
