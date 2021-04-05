# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from typing import List

from src.indexing.models import BaseModel
from src.queries import Query


class RangeQuery(Query):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__(models)

    def predict(self, model_idx, area):
        '''
        Area = (x_min,y_min,x_max,y_max)
        '''
        return self.models[model_idx].range_points(area)

    def range_query(self, area):
        range_point_list = []
        for idx, _ in enumerate(self.models):

            range_point_model = self.predict(idx, area)
            range_point_list.append(range_point_model)
        return range_point_list

    def get_model(self, model_idx: int):
        return self.models[model_idx]
