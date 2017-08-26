# -*- coding: utf-8 -*-

import main.node as Node


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
        new_node = Node.Node()
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

    def __call__(self, node_A: Node.Node, node_B: Node.Node):
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
        assert len(input_val) == 2
        return input_val[0] * input_val[1]

    def gradient(self, node, output_grad):
        """
            返回的分别是 有对于x1的倒数和y对于x2的倒数
        :param node:
        :param output_grad:
        :return:
        """
        return [node.inputs[0] * output_grad, node.inputs[1] * output_grad]


class PlaceHoldOp(Op):
    """
        占位符
    """

    def __call__(self):
        new_node = Op.__call__()
        return new_node


class Executor:
    def __init__(self, eval_node_list):
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        """Computes values of nodes in eval_node_list given computation graph.
        Parameters
        ----------
        feed_dict: list of variable nodes whose values are supplied by user.

        Returns
        -------
        A list of values for nodes in eval_node_list.
        """
        node_to_val_map = dict(feed_dict)
        # Traverse graph in topological sort order and compute values for all nodes.

        topo_order = find_topo_sort(self.eval_node_list)
        for node in topo_order:
            if isinstance(node.op, PlaceHoldOp):
                continue
            vals = [node_to_val_map[n] for n in node.inputs]
            compute_val = node.op.compute(node, vals)
            node_to_val_map[node] = compute_val if isinstance(compute_val, np.ndarray) else np.array(compute_val)

        # Collect node values.
        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results


def find_topo_sort(node_list):
    """Given a list of nodes, return a topological sort list of nodes ending in them.

    A simple algorithm is to do a post-order DFS traversal on the given nodes,
    going backwards based on input edges. Since a node is added to the ordering
    after all its predecessors are traversed due to post-order DFS, we get a topological
    sort.

    """
    visited = set()
    topo_order = []
    for node in node_list:
        topo_sort_dfs(node, visited, topo_order)
    return topo_order


def topo_sort_dfs(node, visited, topo_order):
    """Post-order DFS"""
    if node in visited:
        return
    visited.add(node)
    for n in node.inputs:
        topo_sort_dfs(n, visited, topo_order)
    topo_order.append(node)


def sum_node_list(node_list):
    """Custom sum function in order to avoid create redundant nodes in Python sum implementation."""
    from operator import add
    from functools import reduce
    return reduce(add, node_list)