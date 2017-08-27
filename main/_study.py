import main.autodiff as ad
import numpy as np


def _mul_two_vars():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    y = x2 * x3

    grad_x2, grad_x3 = ad.gradients(y, [x2, x3])

    executor = ad.Executor([y, grad_x2, grad_x3])
    x2_val = 2 * np.ones(3)
    x3_val = 3 * np.ones(3)
    y_val, grad_x2_val, grad_x3_val = executor.run(feed_dict={x2: x2_val, x3: x3_val})

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val * x3_val)
    assert np.array_equal(grad_x2_val, x3_val)
    assert np.array_equal(grad_x3_val, x2_val)


def x2():
    x2 = ad.Variable(name="x2")
    x3 = ad.Variable(name="x3")
    x4 = x2 * x3
    y = x4 * x4
    grad_list = ad.gradients(y, [x2, x3, x4])
    for g in grad_list:
        print("-------21------")
        print(g)


if __name__ == "__main__":
    #_mul_two_vars()
    print("-------------")
    x2()
    pass
