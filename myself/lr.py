# -*- coding: utf-8 -*-

"""
自动微分应用于LR
和soft_max
"""
import codecs
import random
import numpy as np


def get_label_and_feature(path="E:\\CloudMusicProject\\autoDiff\\myself\\x"):
    labels = []
    features = []
    df = []
    with codecs.open(path) as f:
        for line in f:
            lines = line.strip().split("\t")
            label = int(lines[0])
            feature = [float(x) for x in lines[1]]
            if label < 2:
                df.append((label, feature))
    random.shuffle(df)  # 打乱列表
    return df

###################
#      LR         #
###################


class LR(object):

    def __init__(self, x_dim=1):
        self.w = np.random.random((x_dim,)) - 0.5
        self.b = 0
        self.data = None
        pass

    def get_z(self, x):
        z = np.sum(x * self.w) + self.b
        return z

    def get_predict(self, x):
        z = self.get_z(x)
        predict = self.sigmoid(z)
        # print("-------")
        # print(predict)
        return predict

    def get_grad_w_and_b(self, x, y):
        temp = self.get_predict(x) - y
        grad_w = temp * self.get_dz_dw(x)
        grad_b = temp * self.get_dz_db()
        return grad_w, grad_b

    def update_w_b(self, grad_w, grad_b, step_size):
        self.w -= step_size * grad_w
        self.b -= step_size * grad_b

    def print_parm(self):
        print("w:" + str(self.w))
        print("b:" + str(self.b))

    def grad_ceshi(self, x, y, index):
        x = np.array(x)
        print("根据定义算出来的梯度")
        dw = 0.1
        predict_y = self.get_predict(x)
        p = self.get_loss(y, predict_y)
        print(p)
        self.w[index] += dw
        print(self.w[index])
        predict_y_new = self.get_predict(x)
        new_p = self.get_loss(y, predict_y_new)
        print(new_p)
        print(new_p - p)
        dp_dx = (new_p - p) / dw
        print(dp_dx)

        print("根据公式算出来的梯度")
        self.w[index] -= dw
        dw, db = self.get_grad_w_and_b(x, y)
        dp_dx = dw[index]
        print(dp_dx)



    @staticmethod
    def get_loss(y, predict_y):
        if predict_y > 0.999:
            predict_y = 0.999
        elif predict_y < 0.001:
            predict_y = 0.001
        l = y * np.log(predict_y) * (1 - y) + np.log(1 - predict_y)
        return - l

    @staticmethod
    def get_dz_dw(x):
        return np.array(x)

    @staticmethod
    def get_dz_db():
        return 1

    @staticmethod
    def sigmoid(z):
        return 1 / (1 + np.exp(-z))


class TrainLR(object):

    def __init__(self):
        self.lr_model = None
        self.data = None
        self.dim = 0
        pass

    def get_data(self, data):
        self.data = data

    def get_dim(self):
        self.dim = len(self.data[0][1])

    def init_lr_model(self):
        self.lr_model = LR(self.dim)

    def train(self, iter_num, step_size):
        for i in range(iter_num):
            grad_w, grad_b = self.train_step()
            self.lr_model.update_w_b(grad_w, grad_b, step_size)
            pass

    def train_step(self):
        grad_w = np.zeros((self.dim,))
        grad_b = 0.0
        for d in self.data:
            y, x = d
            gw, gb = self.lr_model.get_grad_w_and_b(x, y)
            grad_b -= gb
            grad_w -= gw
        return grad_w, grad_b

    def get_loss(self):
        loss = 0.0
        for d in self.data:
            y, x = d
            y_predict = float(self.lr_model.get_predict(x))
            # print(y)
            # print(self.lr_model.get_loss(y, y_predict))
            loss += self.lr_model.get_loss(y, y_predict)
        return loss
###################
#    softMax      #
###################



###################
#       fm        #
###################


if __name__ == "__main__":
    df = get_label_and_feature()
    tlr = TrainLR()
    tlr.get_data(df)
    tlr.get_dim()
    tlr.init_lr_model()

    print(tlr.lr_model.w)

    y, x = df[0]
    tlr.lr_model.grad_ceshi(x, y, 5)

    # tlr.train(1, 0.5)
    # print(tlr.get_loss())
    #
    # tlr.train(1, 0.5)
    # print(tlr.get_loss())

    pass


