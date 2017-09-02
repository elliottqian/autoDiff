# -*- coding: utf-8 -*-

"""
绘制点图和直线图
"""

import numpy as np
import matplotlib.pyplot as plt

x = np.arange(1,10)

y = x
z = -x

# fig = plt.figure()
#
# ax1 = fig.add_subplot(111)
#
# # 设置标题
# ax1.set_title('Scatter Plot')
# # 设置X轴标签
# plt.xlabel('X')
# # 设置Y轴标签
# plt.ylabel('Y')
# # 画散点图
# ax1.scatter(x, y, c='r', marker='o')
# ax1.scatter(x, z, c='b', marker='o')
# # 设置图标
# plt.legend('x1')
# # 显示所画的图
# plt.show()
#
#
# def create():
#     fig = plt.figure()
#     ax1 = fig.add_subplot(111)
#     return fig, ax1
#
# def plot_point(ax, x, y, color):
#     ax.scatter(x, y, c=color, marker='o')
#
# def plot_line(x ,y):
#     pass





def name2type(color, line_type):
    """

    :param color: 颜色: 主要有; green, yellow, red
    :param line_type: 线段类型: 主要有 point, line
    :return:
    """
    c = ""
    t = ""
    if color == "green":
        c = "g"
    elif color == "yellow":
        c = "g"
    else:
        c = "r"

    if line_type == "point":
        t = "."
    elif line_type == "line":
        t = "-"
    else:
        t = "-"
    xxx = c + t
    return xxx

ty = name2type("green", "point")


plt.figure('data')

plt.plot(x, y, ty)
plt.plot(x, z, name2type("red", "line"))
plt.show()