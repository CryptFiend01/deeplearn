import numpy as np
import h5py
import matplotlib.pyplot as plt
import scipy
from scipy import ndimage
#import skimage
#import imageio
from PIL import Image
from logist import logist_model, logist_predict
from nn import nn_model
from deep_nn import dnn_model, dnn_predict
from utils import saveParams, loadParams

def load_h5_data():
    train_dataset = h5py.File('ml/datasets/train_catvnoncat.h5', 'r')
    train_X = np.array(train_dataset['train_set_x'][:])
    train_Y = np.array(train_dataset['train_set_y'][:])
    
    train_Y = train_Y.reshape((1, train_Y.shape[0]))
    test_dataset = h5py.File("ml/datasets/test_catvnoncat.h5", 'r')
    test_X = np.array(test_dataset['test_set_x'][:])
    test_Y = np.array(test_dataset['test_set_y'][:])
    test_Y = test_Y.reshape((1, test_Y.shape[0]))
    classes = np.array(test_dataset['list_classes'][:])
    return train_X, train_Y, test_X, test_Y, classes

def load_image(image_name):
    image = np.array(plt.imread(image_name))
    trans_image = scipy.misc.imresize(image, size=(64, 64))
    return image, trans_image.reshape((1, 64*64*3)).T

def retrain_logist():
    X_train, Y_train, X_test, Y_test, _ = load_h5_data()
    X_train = X_train.reshape((X_train.shape[0], -1)).T / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).T / 255
    print("X_shape:" + str(X_train.shape))
    print("Y_shape:" + str(Y_train.shape))
    d = logist_model(X_train, Y_train, X_test, Y_test, 4000, 0.005, True)

    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iteration times")
    plt.show()
    
    f = open("wb.txt", 'w')
    W = d['w'].tolist()
    b = d['b']
    ws = []
    for w in W:
        ws.append(str(w[0]))
    s = "%s;%s" % (','.join(ws), str(b))
    f.write(s)
    f.close()

def use_logist():
    f = open('wb.txt', 'r')
    s = f.read()
    p = s.split(';')
    wl = []
    ws = p[0].split(',')
    for q in ws:
        wl.append(float(q))
    w = np.array(wl).reshape((64*64*3, 1))
    b = float(p[1])

    image, dataset = load_image("ml/image/gou5.jpg")
    plt.imshow(image)
    predict = logist_predict(w, b, dataset)
    if predict[0, 0] == 1:
        print("It's a cat.")
    else:
        print("It's not a cat")
    plt.show()

def retrain_nn():
    X_train, Y_train, X_test, Y_test, _ = load_h5_data()
    X_train = X_train.reshape((X_train.shape[0], -1)).T / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).T / 255
    print("X_shape:" + str(X_train.shape))
    print("Y_shape:" + str(Y_train.shape))

    d = nn_model(X_train, Y_train, X_test, Y_test, 7, 2500, 0.05, True)

    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iteration times")
    plt.show()

def retrain_dnn():
    X_train, Y_train, X_test, Y_test, _ = load_h5_data()
    X_train = X_train.reshape((X_train.shape[0], -1)).T / 255
    X_test = X_test.reshape((X_test.shape[0], -1)).T / 255
    print("X_shape:" + str(X_train.shape))
    print("Y_shape:" + str(Y_train.shape))
    np.random.seed(1)
    d = dnn_model(X_train, Y_train, X_test, Y_test, [20, 7, 5], 2500, 0.008, True, 0, 1)
    costs = np.squeeze(d['costs'])
    plt.plot(costs)
    plt.ylabel("cost")
    plt.xlabel("iteration times")
    plt.show()

    # saveParams(d, 'dnn.h5')

def use_dnn():
    params = loadParams('dnn.h5')
    image, dataset = load_image("ml/image/gou2.jpg")
    p = dnn_predict(params, dataset)
    print(p)

def saveImage():
    X_train, Y_train, X_test, Y_test, _ = load_h5_data()
    #print(X_train[0].shape)
    for i in range(len(X_test)):
        plt.imsave('ml/test/%d.jpg' % (i + 1), X_test[i])

if __name__ == '__main__':
    #retrain_logist()
    #use_logist()
    retrain_dnn()
    #use_dnn()
    #saveImage()
