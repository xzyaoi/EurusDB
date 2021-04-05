# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import numpy as np

from src.indexing.learning.polynomial_regression import PolynomialRegression

X = np.array([1, 2, 3, 4, 5, 6])
y = np.array([1, 4, 9, 16, 25, 36])
y = y.reshape((len(y)))

print(y.shape)

pr = PolynomialRegression(2)

pr.fit(X, y)

y_pred = pr.predict(np.array([1]))
