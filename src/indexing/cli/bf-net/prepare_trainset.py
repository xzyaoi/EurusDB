# In this file, we use pwlf to fit continuous piecewise linear functions
# Ramer-Douglas-Peucker Algorithm used in this script
# Reference:
# https://en.wikipedia.org/wiki/Ramer%E2%80%93Douglas%E2%80%93Peucker_algorithm

import sys
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.indexing.utilities.dataloaders import normalize


def read_data(filename):
    df = pd.read_csv(filename).values
    X = df[0:, 0]
    Y = df[0:, 1]
    X = normalize(X)
    Y = normalize(Y)
    return X, Y


def distance(X, point, line: Tuple):
    p3 = np.array([point, X[point]])
    p1 = np.array([line[0], X[line[0]]])
    p2 = np.array([line[1], X[line[1]]])
    d = abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
    return d


def DouglasPeucker(X, epsilon):
    dmax = 0
    index = 0
    end = len(X) - 1

    for i in range(end - 2):
        d = distance(X, i + 1, (0, end))
        if (d > dmax):
            index = i + 1
            dmax = d
    results = []
    if (dmax > epsilon):
        recResults1 = DouglasPeucker(X[0:index], epsilon)
        recResults2 = DouglasPeucker(X[index:end], epsilon)
        results = recResults1 + recResults2
    else:
        results = [X[0], X[end]]

    return results


def visualize(results, X, Y):
    plt.scatter(X, Y, s=np.pi * 3, alpha=0.5)
    plt.title('x-y of the lognormal')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.vlines(results, colors='blue', ymin=0, ymax=1)
    plt.show()


if __name__ == "__main__":
    filename = sys.argv[1]
    X, Y = read_data(filename)
    results = DouglasPeucker(X, 0.2)
    print(results)
    print("Number of Breakpoints: {}".format(len(results)))
    visualize(results, X, Y)
