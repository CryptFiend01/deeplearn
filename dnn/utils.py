import numpy as np
import h5py

def sigmoid(z):
    return 1 / ( 1 + np.exp(-z) )

def relu(z):
    return np.maximum(0, z)

def accuracy(Y, Y_prediction):
    return 100 - np.mean(np.abs(Y - Y_prediction)) * 100

def makeZ(W, X, b):
    # W.shape = (n_x1, n_x2)
    # X.shape = (n_x1, m)
    # b numeric or b.shape = (n_x2, 1)
    return np.dot(W.T, X) + b

def saveParams(params, fname):
    f = h5py.File(fname, 'w')
    for k, v in params.items():
        if k[0] == 'W' or k[0] == 'b':
            f.create_dataset(k, data=v)

def loadParams(fname):
    f = h5py.File(fname, 'r')
    params = {}
    for k in f.keys():
        params[k] = f[k].value
    return params

if __name__ == '__main__':
    print(sigmoid(np.array([-3497093.08947368,  -627652.12954545, -2756031.08421053,  -929172.56483254])))