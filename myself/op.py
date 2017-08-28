# -*- coding: utf-8 -*-

import sys
sys.path.append("E:\\CloudMusicProject\\autoDiff")

from myself.node import Node, ones_like_op, PlaceholderOp
import numpy as np



def gradients(target, x_list):
    """

    :param target:  求导目标, 一般这里就是损失函数
    :param x_list:  对x求偏微分, 这里对x_list里面的所有节点求偏导数
    :return:        x_list 对应的偏导数的节点
    """
    # 每个节点对应的输出的那啥  后面再理清楚 每个节点依赖节点的倒数,
    node_to_output_grads_list = {}
    output_node = target
    # 输出节点一般就是y, y对y求微分自然是1, list 是为了统一格式, 因为以后某个节点可能有多个输出
    node_to_output_grads_list[output_node] = [ones_like_op(output_node)]
    # 每个节点对应的微分节点, 这个就是最终的计算结果了, 把依赖的节点的导数reduce一下  求和  就是最终的结果了
    node_to_output_grad = {}

    reverse_topo_order = reversed(find_topo_sort_qw([output_node]))

    for node in reverse_topo_order:
        """
        # 这段思路不记得了
        """
        grad = sum_node_list(node_to_output_grads_list[node])
        node_to_output_grad[node] = grad
        grads = node.op.gradient(node, grad)
        for i in range(len(node.inputs)):
            temp_node = node.inputs[i]
            grads_list = node_to_output_grads_list.get(temp_node, [])
            grads_list.append(grads[i])
            node_to_output_grads_list[temp_node] = grads_list

    grad_node_list = [node_to_output_grad[node] for node in x_list]
    return grad_node_list


def find_topo_sort_qw(node_list):
    """
    拓扑排序, 采用深度优先, 后续遍历的方式, 输入某个节点, 然后从这个节点的叶子节点开始遍历
    :return:
    """
    has_visted = set()
    topo_order = []
    for node in node_list:
        print(node)
        # 这里有一个合并的思想, 把所有的节点合并到一个拓扑排序中
        topo_sort_dfs_qw(node, has_visted, topo_order)
    return topo_order


def topo_sort_dfs_qw(node, has_visted, topo_order):
    """

    :param node:
    :type node: Node
    :param has_visted:
    :param topo_order:
    :return:
    """
    if node in has_visted:
        return
    has_visted.add(node)
    for nod in node.inputs:
        topo_sort_dfs_qw(nod, has_visted, topo_order)
    topo_order.append(node)


class Executor:

    def __init__(self, eval_node_list):
        """
        传入需要计算出值的节点
        :param eval_node_list:
        """
        self.eval_node_list = eval_node_list

    def run(self, feed_dict):
        node_to_val_map = dict(feed_dict)
        topo_order = find_topo_sort_qw(self.eval_node_list)

        for node in topo_order:
            if isinstance(node.op, PlaceholderOp):
                continue
            vals = [node_to_val_map[n] for n in node.inputs]  # val 是从第二层计算节点中 索引到输入节点
            compute_val = node.op.compute(node, vals)         # 再通过这些输入节点, 计算当前节点的值
            # 保存当前节点的值, 方便下一层继续索引
            node_to_val_map[node] = compute_val if isinstance(compute_val, np.ndarray) else np.array(compute_val)

        node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
        return node_val_results
# class Executor:
#     def __init__(self, eval_node_list):
#         self.eval_node_list = eval_node_list
#
#     def run(self, feed_dict):
#         """Computes values of nodes in eval_node_list given computation graph.
#         Parameters
#         ----------
#         feed_dict: list of variable nodes whose values are supplied by user.
#
#         Returns
#         -------
#         A list of values for nodes in eval_node_list.
#         """
#         node_to_val_map = dict(feed_dict)
#         # Traverse graph in topological sort order and compute values for all nodes.
#
#         topo_order = find_topo_sort(self.eval_node_list)
#         for node in topo_order:
#             if isinstance(node.op, PlaceHoldOp):
#                 continue
#             vals = [node_to_val_map[n] for n in node.inputs]
#             compute_val = node.op.compute(node, vals)
#             node_to_val_map[node] = compute_val if isinstance(compute_val, np.ndarray) else np.array(compute_val)
#
#         # Collect node values.
#         node_val_results = [node_to_val_map[node] for node in self.eval_node_list]
#         return node_val_results


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