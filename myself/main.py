# -*- coding: utf-8 -*-

import numpy as np
from myself.node import variable
from myself.op import find_topo_sort_qw, gradients, Executor

if __name__ == "__main__":
    x1 = variable("x1")
    x2 = variable("x2")
    y = x1.mul_other_node(x2)
    print(y)

    print("""开始构造求导图, 测试拓扑排序""")
    top_list = find_topo_sort_qw([y])
    print(len(top_list))
    for node in top_list:
        print(node)

    print("""开始构造求导图, 测试拓扑排序""")
    r = gradients(y, [x1, x2])
    for x in r:
        print(x)

    print("开始执行, Executor")
    x1_eval = np.array(3)
    x2_eval = np.array(4)
    e = Executor(r)
    # 这里的r2就是梯度了
    r2 = e.run(feed_dict={x1: x1_eval, x2: x2_eval})
    print(r2)

    pass
