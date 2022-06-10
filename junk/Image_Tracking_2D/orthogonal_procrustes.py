import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as sp
from timeit import timeit


def main():
    data0 = np.load("obj.npy")
    data1 = np.load("obj2.npy")

    smaller = min([len(data1), len(data0)])

    first = data0[::len(data0) // smaller + 1]
    second = data1[::len(data1) // smaller]

    s2 = min([len(first), len(second)])
    mat = sp.orthogonal_procrustes(first[:s2], second[:s2])[0]
    print(mat)
    # new = np.matmul(data0, np.linalg.inv(mat))

    length = data1.shape[0]
    sum_x = np.sum(data1[:, 0])
    sum_y = np.sum(data1[:, 1])
    cent_x, cent_y = sum_x / length, sum_y / length

    plt.ylim(0, 360)
    plt.xlim(0, 640)
    # plt.plot(data0[:, 0], data0[:, 1], 'b.')
    plt.plot(data1[:, 0], data1[:, 1], 'r.')
    plt.plot([cent_x], [cent_y], 'bo')
    # plt.plot(new[:, 0], new[:, 1], 'g.')
    plt.show()


if __name__ == "__main__":
    main()
    # data = []
    # for i in range(100):
    #     t = timeit("main()", globals=globals(), number=1)
    #     data.append(t)
    #
    # avg = np.average(data)
    # std = np.std(data)
    # print(avg)
    # print(std)
