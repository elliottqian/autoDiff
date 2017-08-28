# -*- coding: utf-8 -*-

import numpy as np

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

    def mul_other_node(self, other):
        """
        和另一个节点相乘
        :param other:
        :return:
        """
        if isinstance(other, Node):
            new_node = mul_op(self, other)
            return new_node
        else:
            return None

    def add_other_node(self, other):
        if isinstance(other, Node):
            new_node = mul_op(self, other)
            return new_node
        else:
            return None

    def __str__(self):
        return self.name


def variable(name):
    """User defined variables in an expression.
        e.g. x = Variable(name = "x")
    """
    placeholder_node = placeholder_op()  # 调用 call 函数
    placeholder_node.name = name
    return placeholder_node


class Op(object):
    """
        Op represents operations performed on nodes.
        整个工程的思路是, 保存每个节点的梯度值, 然后重复调用
        有点类似于先化简再计算
    """

    def __call__(self):
        """Create a new node and associate the op object with the node.

        Returns
        -------
        The new node object.
        """
        new_node = Node()
        new_node.op = self
        return new_node

    def compute(self, node, input_val):
        """
            计算当前节点输出值
        :param node:        当前节点
        :param input_val:   输入值
        :return:
        """
        assert False, "在子类中实现"

    def gradient(self, node, output_grad):
        """
            通过子节点, 子节点又指什么, 接近y的节点吗
            返回梯度值
        :param node:
        :param output_grad:
        :return:
        """
        assert False, "在子类中实现"



class MulOp(Op):
    """
        乘法操作
    """

    def __call__(self, node_A, node_B):
        """

        :param node_A:
        :param node_B:
        :return:
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "%s * %s" %(node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_val):
        """
        节点和对应的输入节点的值, 只有数值节点, 没有操作节点, 操作都是在数值节点上的
        :param node:
        :param input_val:
        :return:
        """
        assert len(input_val) == 2
        return input_val[0] * input_val[1]

    def gradient(self, node, output_grad):
        """
            返回的分别是 有对于x1的倒数和y对于x2的倒数
            因为乘法符被重载了, 所以返回的是节点
            现在没有弄清楚的是feed 怎么完成 符号到数值的转变, 也是从图的起始节点一个一个迭代的吗
        :param node:
        :param output_grad:
        :return:
        """
        return [node.inputs[1].mul_other_node(output_grad), node.inputs[0].mul_other_node(output_grad)]

mul_op = MulOp()


class AddOp(Op):

    def __call__(self, node_A, node_B):
        """

        :type node_A: Node
        :type node_B: Node
        :return:
        """
        new_node = Op.__call__(self)
        new_node.inputs = [node_A, node_B]
        new_node.name = "%s + %s" % (node_A.name, node_B.name)
        return new_node

    def compute(self, node, input_val):
        """
        :param node:
        :type input_val: list[int]
        :return:
        """
        assert len(input_val) == 2
        return input_val[0] + input_val[1]

    def gradient(self, node, output_grad):
        """
        加法的导数是1, 所以就直接传递就好了
        :param node:
        :param output_grad:
        :return:
        """
        return [output_grad, output_grad]

class PlaceholderOp(Op):
    """
        Op to feed value to a nodes.
        求职运算从Placeholder节点开始, Placeholder节点就是操作符为PlaceholderOp的节点
    """
    def __call__(self):
        """
            Creates a variable node.
            返回一个节点, 然后传入自己给节点的op变量
        """
        new_node = Op.__call__(self)
        return new_node

    def compute(self, node, input_vals):
        """No compute function since node value is fed directly in Executor."""
        assert False, "placeholder values provided by feed_dict"

    def gradient(self, node, output_grad):
        """No gradient function since node has no inputs."""
        return None

placeholder_op = PlaceholderOp()


class OnesLikeOp(Op):
    """
        Op that represents a constant np.ones_like.
    """
    def __call__(self, node_A):
        """Creates a node that represents a np.ones array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Oneslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns ones_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.ones(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

ones_like_op = OnesLikeOp()

"""
这里这些操作是不是单例模式呢?
"""
class ZerosLikeOp(Op):
    """Op that represents a constant np.zeros_like."""
    def __call__(self, node_A):
        """Creates a node that represents a np.zeros array of same shape as node_A."""
        new_node = Op.__call__(self)
        new_node.inputs = [node_A]
        new_node.name = "Zeroslike(%s)" % node_A.name
        return new_node

    def compute(self, node, input_vals):
        """Returns zeros_like of the same shape as input."""
        assert(isinstance(input_vals[0], np.ndarray))
        return np.zeros(input_vals[0].shape)

    def gradient(self, node, output_grad):
        return [zeroslike_op(node.inputs[0])]

zeroslike_op = ZerosLikeOp()

if __name__ == "__main__":
    pass