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
    sum = 0.0
    m, n = v.shape
    for i in range(m):
        for j in range(i + 1, m):
            # print(v[i] * v[j])
            #print(x[i] * x[j])
            temp = x[i] * x[j] * np.sum(v[i] * v[j])
            # print(temp)
            sum += temp
    z = np.sum(x * w) + w0 + sum
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
    step_size = 0.1
    loss_ = 0.0
    w0 = - 0.5
    w = np.random.random((len(features[0]),)) - 0.5
    print(w)
    print(w.shape)
    v = np.random.random((len(features[0]), 3)) - 0.5
    print(v.shape)
    for iter in range(max_iter):
        loss_ = 0.0
        for index_x in range(len(features)):
            y = labels[index_x]
            x = np.array(features[index_x])
            z = get_z(w0, w, v, x)
            #print("-----------")
            #print(z)
            pred_y = sigmoid(z)
            #print(pred_y)
            loss_ += loss(y, pred_y)
            #print(sigmoid(z))
            #print(grad_sigmoid(sigmoid(z)))

            #print(type(pred_y))
            #print(type(y))
            temp = y - pred_y
            #print(temp)

            w0 = w0 + step_size * temp  * grad_dz_dw0()
            #print(step_size * temp  * grad_dz_dw0())
            #print(w0)
            #print("w0:" + str(w0))
            w = w + step_size * temp * grad_dz_dw(x)
            v = v + step_size * temp * grad_dz_dv_mat_main(x, v)
        print(w0)
        print(loss_ / len(features))


def ceshi_grad(w0, w, v, x):


    d = 0.01
    z = get_z(w0, w, v, x)
    print(z)
    print("v的导数")
    v[0][0] = v[0][0] + d
    print(v)
    print("更改v后的z")
    z_ = get_z(w0, w, v, x)
    print(z_)
    dz = z_ - z
    dz_dv00 = dz / d
    print("根据定义的导数")
    print(dz_dv00)
    v[0][0] = v[0][0] - d
    print(grad_dz_dv_mat_main(x, v))

    print("看下 w0 的导数")
    print(grad_dz_dw0())

    v[0, 0] = v[0, 0] - d
    w0 = w0 + d
    z_w0_ = get_z(w0, w, v, x)
    dz_w0 = z_w0_ - z
    print(dz_w0/d)

    print("看下 w 的导数")
    print(grad_dz_dw(x))
    z = get_z(w0, w, v, x)
    print(z)
    w = w[1] + d
    z_w = get_z(w0, w, v, x)
    print(z_w)
    dz_w = z_w - z
    print(dz_w)
    print(dz_w/d)


def w_grad_ceshi(w0, w, v, x):
    d = 0.05

    print("公式计算w[1]的导数")
    print(grad_dz_dw(x))

    print("计算z")
    z = get_z(w0, w, v, x)
    print("z")
    print("更新w[1]后的z")
    w[1] = w[1] + d
    print(v)
    new_z = get_z(w0, w, v, x)
    print("根据定义算的导数")
    print((new_z - z)/d)


def v_grad_ceshi(w0, w, v, x):
    d = 1

    print("公式计算v[1]的导数")
    print(grad_dz_dv_mat_main(x, v))

    print("计算z")
    z = get_z(w0, w, v, x)
    print(z)
    print("更新w[1]后的z")
    v[1][1] = v[1][1] + d

    new_z = get_z(w0, w, v, x)
    print(new_z)
    print("根据定义算的导数")
    print((new_z - z)/d)

if __name__ == "__main__":
    x_example = np.array([1.0, 2, 3])
    y_example = 0

    w0_example = 0.5
    w_example = np.array([1.0, 2, 3])
    v_example = np.array([[1.0, 3], [2, 2], [3, 1]])

    # print(get_z(w0_example, w_example, v_example, x_example))
    # w_grad_ceshi(w0_example, w_example, v_example, x_example)
    v_grad_ceshi(w0_example, w_example, v_example, x_example)
    # def p(x, v):
    #     return x.dot(v).dot(v.T).dot(x.T)

    # print(np.sum(x_example * w_example) + w0_example + x_example.dot(v_example).dot(v_example.T).dot(x_example.T))

    # print(p(x_example, v_example))

    # print(grad_dz_dv_mat_main(x_example, v_example))

    # print(loss(1, 1))

    # ceshi_grad(w0_example, w_example, v_example, x_example)

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
    # print("grad_dz_dv_mat")
    # dv = grad_dz_dv_mat(x_example, v_example)
    # print(dv)




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
