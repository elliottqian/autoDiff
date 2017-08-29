# -*- coding: utf-8 -*-

import numpy as np


def get_z(w0, w, v, x):
    """

    :param w0: 常数
    :param w:  向量
    :type w:  np.array
    :param v:  矩阵
    :param x:  向量
    :return:
    """
    z = np.sum(x * w) + w0 + x.dot(v).dot(v.T).dot(x.T)
    return z


def sigmoid(z):
    return 1 / (1 + np.exp(-z))


def loss(y, y_predict):
    if int(y_predict) == 0:
        y_predict = 0.00001
    elif int(y_predict) == 1:
        y_predict = 0.99999
    l = y * np.log(y_predict) + (1 - y) * np.log(1 - y_predict)
    return -l


def grad_dz_dw0():
    return 1


def grad_dz_dw(x):
    return x


def grad_dz_dv(x, v, index, f):
    m, n = v.shape
    temp = 0.0
    for j in range(m):
        temp += v[j, f] * x[j]
    return x[index] * temp - v[index, f] * np.power(x[index], 2)


def grad_dz_dv_mat(x, v):
    grad_ = np.zeros_like(v)
    m, n = v.shape
    for index in range(m):
        for f in range(n):
            grad_[index, f] = grad_dz_dv(x, v, index, f)
    return grad_


def grad_sigmoid(sigmoid_):
    return sigmoid_ * (1 - sigmoid_)


def grad_dz_dv_mat_main(x, v):
    m, n = v.shape
    a = v * x.reshape((m, 1))
    b = np.sum(a, 0)
    c = x.reshape((m, 1)).dot(b.reshape((1, n)))
    x_2 = x * x
    t_ = v * x_2.reshape((m, 1))
    r = c - t_
    return r


def gradient_s(max_iter, features, labels):
    step_size = 1
    loss_ = 0.0
    w0 = 0.5
    w = np.random.random((len(features[0]),)) - 0.5
    print(w.shape)
    v = np.random.random((len(features[0]), 3)) - 0.5
    print(v.shape)
    for iter in range(max_iter):
        loss_ = 0.0
        for index_x in range(len(features)):
            y = labels[index_x]
            x = np.array(features[index_x])
            z = get_z(w0, w, v, x)
            pred_y = sigmoid(z)
            loss_ += loss(y, pred_y)
            #print(sigmoid(z))
            #print(grad_sigmoid(sigmoid(z)))
            w0 = w0 - step_size * grad_sigmoid(sigmoid(z)) * grad_dz_dw0()
            #print("w0:" + str(w0))
            w = w - step_size * grad_sigmoid(sigmoid(z)) * grad_dz_dw(x)
            v = v - step_size * grad_sigmoid(sigmoid(z)) * grad_dz_dv_mat_main(x, v)
        print(w0)
        print(loss_ / len(features))

if __name__ == "__main__":
    x_example = np.array([1, 2, 3])
    y_example = 0

    w0_example = 0.5
    w_example = np.array([1, 2, 3])
    v_example = np.ones((3, 2))
    v_example[1, 1] = 2

    print(grad_dz_dv_mat_main(x_example, v_example))

    print(loss(1, 1))

    # print("get_z")
    # a = get_z(w0_example, w_example, v_example, x_example)
    # print(a)
    #
    # print("sigmoid")
    # y_predict_ = sigmoid(a)
    # print(y_predict_)
    #
    # print("loss")
    # l_ = loss(y_example, y_predict_)
    # print(l_)
    #
    # print("grad_dz_dw")
    # dw = grad_dz_dw(x_example)
    # print(dw)
    #
    print("grad_dz_dv_mat")
    dv = grad_dz_dv_mat(x_example, v_example)
    print(dv)




    #
    #
    # print()
    # x_length = 10
    # k = 2
    # w0 = 0
    #
    # v = np.ones((x_length, 2))
    # m ,n = v.shape
    # print(m, n)
    # print(x_simple.reshape((1, 10)))
    # print(x_simple.reshape((1, 10)).dot(v).dot(v.T).dot(x_simple.reshape(1, 10).T))
    #
    # z = np.sum(x_simple * w) + w0 + x_simple.dot(v).dot(v.T).dot(x_simple.T)
    #
    # print(z)
    # print(1 / (1 + np.exp(-z)))
    #
    #
    # print(np.asmatrix(1))
    pass
