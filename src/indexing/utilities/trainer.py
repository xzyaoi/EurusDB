# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List

from src.indexing.models import BaseModel
from src.indexing.utilities.dataloaders import split_train_test


class ModelTrainer():
    def __init__(self, models: List[BaseModel]) -> None:
        self.models = models
        self.debug = False

    def build(self, data, test_ratio, use_index=True, sample_ratio=1):
        self.sample_ratio = sample_ratio
        if use_index:
            x_train, _, x_test, y_test, y_train = split_train_test(
                data, test_ratio)
        else:
            x_train, y_train, x_test, y_test, _ = split_train_test(
                data, test_ratio)
        return self._build(x_train, y_train, x_test, y_test)

    def _build(self, x_train, y_train, x_test, y_test):
        build_times = []
        mses = []
        for model in self.models:
            mse, build_time = model.train(x_train, y_train, x_test, y_test)
            print("{} model built in {:.4f} ms, mse={:4f}".format(
                model.name, build_time * 1000, mse))
            build_times.append(build_time)
            mses.append(mse)
        return mses, build_times

    def get_models(self):
        return self.models
