# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import json
import uuid
from typing import Dict


class BaseModel(object):
    def __init__(self, name, savepath=None) -> None:
        super().__init__()
        self.name = name
        self.id = uuid.uuid4().__str__()[0:8]
        self.savepath = savepath

    def train(self, x_train, y_train, x_test, y_test) -> None:
        # x, y are numpy array
        raise NotImplementedError

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
