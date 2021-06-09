# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import uuid
from typing import Dict

import numpy as np

from src.indexing.utilities.dataloaders import normalize


class BaseModel(object):
    def __init__(self, name, savepath=None) -> None:
        super().__init__()
        self.name = name
        self.id = uuid.uuid4().__str__()[0:8]
        self.savepath = savepath

    def train(self, x_train, y_train, x_test, y_test) -> None:
        # x, y are numpy array
        raise NotImplementedError

    def _normalize(self, x_train, y_train, x_test, y_test):
        if not self.has_normalized:
            self.max_x = np.max(x_train)
            self.min_x = np.min(x_train)
            self.max_y = np.max(y_train)
            self.min_y = np.min(y_train)
            x_train, y_train = normalize(x_train), normalize(y_train)

            if x_test is not None:
                x_test = (x_test - self.min_x) / (self.max_x - self.min_x + 1)
            if y_test is not None:
                y_test = (y_test - self.min_y) / (self.max_y - self.min_y + 1)
            return x_train, y_train, x_test, y_test
        else:
            print("has already normalized...")

    def predict(self, x):
        raise NotImplementedError

    def export(self) -> Dict:
        raise NotImplementedError

    def save(self, path: str = None, comments: Dict = None):
        model_dict = self.export()
        model_dict["meta"] = {"name": self.name, "id": self.id}
        model_dict["comments"] = comments
        if path is not None:
            with open(path, 'w') as outfile:
                json.dump(model_dict, outfile)
        else:
            if self.savepath is None:
                raise ValueError("No save path specified")
            else:
                with open(self.savepath, 'w') as outfile:
                    json.dump(model_dict, outfile)
