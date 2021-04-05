# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from src.indexing.learning.piecewise import PiecewiseRegression
from src.indexing.models import BaseModel


class ConvModel(BaseModel):
    def __init__(self, name, page_size) -> None:
        super().__init__(name, page_size=page_size)

    def train(self, x_train, y_train, x_test, y_test):
        pass
