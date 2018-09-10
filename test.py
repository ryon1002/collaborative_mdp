import numpy as np
# from tiger.pomdp_to import Tiger
from tiger.pomdp_to2 import Tiger
# from tiger.pomdp_to3 import Tiger

if __name__ == '__main__':
    import matplotlib.pyplot as plt

    b1 = np.arange(0, 1.01, 0.04)
    b2 = 1 - b1
    b3 = np.zeros_like(b1)
    b = np.concatenate(([b1], [b2], [b3]), axis=0).T
    t = Tiger()
    for d in [1, 2, 3, 6, 30]:
    # for d in [1, 2, 3, 6]:
    # for d in [1, 2]:
        t.calc_a_vector(d, b, with_a=False)
        v = np.array([t.value(b[i]) for i in range(len(b))])
        plt.plot(b[:, 0], v)
    plt.show()
