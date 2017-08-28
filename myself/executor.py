import numpy as np

from myself.op import PlaceHoldOp, find_topo_sort


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
