import main.autodiff as ad
import numpy as np


def __identity():

    x2 = ad.Variable(name="x2")
    y = x2

    grad_x2, = ad.gradients(y, [x2])

    executor = ad.Executor([y, grad_x2])
    x2_val = 2 * np.ones(3)
    print(x2_val)
    y_val, grad_x2_val = executor.run(feed_dict={x2: x2_val})

    print(y_val, grad_x2_val)

    assert isinstance(y, ad.Node)
    assert np.array_equal(y_val, x2_val)
    assert np.array_equal(grad_x2_val, np.ones_like(x2_val))


if __name__ == "__main__":
    __identity()
    pass
