# -*- coding: utf-8 -*-


class Node(object):
    """
        节点可以分为3类
        常数节点;
            const_attr代表节点的常数值
        变量节点:
        操作符节点:
            op成员用于存储节点的操作算子
        name是节点的标识
    """
    def __init__(self):
        self.inputs = []
        self.op = None
        self.const_attr = None
        self.name = ""

if __name__ == "__main__":
    pass