from typing import Tuple

import numpy as np


class RDPAlgorithm():
    def __init__(self, epsilon) -> None:
        self.epsilon = epsilon

    def douglasPeucker(self, X):
        dmax = 0
        index = 0
        end = len(X) - 1
        for i in range(end - 2):
            d = self.distance(X, i + 1, (0, end))
            if (d > dmax):
                index = i + 1
                dmax = d
        results = []
        if (dmax > self.epsilon):
            recursiveResultsLeft = self.douglasPeucker(X[0:index])
            recursiveResultsRight = self.douglasPeucker(X[index:end])
            results = recursiveResultsLeft + recursiveResultsRight
        else:
            results = [X[0], X[end]]
        return results

    def fit(self, X):
        return self.douglasPeucker(X)

    def distance(self, X, point, line: Tuple):
        p3 = np.array([point, X[point]])
        p1 = np.array([line[0], X[line[0]]])
        p2 = np.array([line[1], X[line[1]]])
        return abs(np.cross(p2 - p1, p3 - p1) / np.linalg.norm(p2 - p1))
