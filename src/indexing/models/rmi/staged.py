# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
This file describes how staged model, i.e. recursive model works.

@author: Xiaozhe Yao
@updated: 14. Mar. 2021.
'''

import sys
from timeit import default_timer as timer
from typing import List

import numpy as np

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.indexing.models.ml.polynomial_regression import PolynomialRegression
from src.indexing.models.nn.fcn import FCNModel
from src.indexing.models.trees.b_tree import BTreeModel

class StagedModel(BaseModel):
    def __init__(self, model_types, num_models, page_size) -> None:
        super().__init__("Staged Model", page_size)
        self.num_of_stages = len(model_types)
        self.num_of_models = num_models
        if not len(self.num_of_models) == self.num_of_stages:
            raise ValueError(
                "len(num_models) is expected to be equal to len(model_types)")
        self.models: List = []
        self.model_types = model_types

    def _build_single_model(self, model_type, train_data):
        x_train = train_data[0]
        y_train = train_data[1]
        if model_type == 'lr':
            model = PolynomialRegression(1)
        elif model_type == 'quadratic':
            model = PolynomialRegression(2)
        elif model_type == 'b-tree':
            model = BTreeModel(page_size=self.page_size, degree=10)
        elif model_type == 'fcn':
            model = FCNModel(layers=[1, 4, 4, 1],
                             activations=['relu', 'relu', 'relu'],
                             epochs=2000,
                             page_size=self.page_size,
                             lr=0.001)
        else:
            raise ValueError("Unsupported Model Type")
        model.fit(x_train, y_train)
        return model

    def train(self, x_train, y_train, x_test, y_test):
        self.max_pos = np.max(y_train)
        train_data = (x_train, y_train)
        # a 2-d array indexed by [stage][model_id]
        train_datas = [[train_data]]
        start_time = timer()
        for stage in range(self.num_of_stages):
            number_unused_model = 0
            self.models.append([])
            for model_id in range(self.num_of_models[stage]):
                if train_datas[stage][model_id][0] is not None:
                    model = self._build_single_model(
                        self.model_types[stage], train_datas[stage][model_id])
                    self.models[stage].append(model)
                else:
                    self.models[stage].append(None)
            if not stage == self.num_of_stages - 1:
                # if it is not the last stage
                # prepare dataset for the next stage
                # the next_xs and next_ys are two dimensional list
                # indexed by stage_id, model_id
                next_xs = [[] for i in range(self.num_of_models[stage + 1])]
                next_ys = [[] for i in range(self.num_of_models[stage + 1])]
                for index, key in enumerate(x_train):
                    model_id = self.get_staged_output(key, stage)
                    output = self.models[stage][model_id].predict(key)
                    selected_model_id = int(
                        output * self.num_of_models[stage + 1] / self.max_pos)
                    # in case selected_model_id is not in range
                    selected_model_id = self.acceptable_next_model(
                        selected_model_id, stage)
                    # print('selected model id: {}'.format(selected_model_id))
                    next_xs[selected_model_id].append(key)
                    next_ys[selected_model_id].append(y_train[index])

                # prepare data accordingly
                for next_model_id in range(self.num_of_models[stage + 1]):
                    train_datas.append([])
                    if len(next_xs[next_model_id]) != 0:
                        dataset = (next_xs[next_model_id],
                                   next_ys[next_model_id])
                        train_datas[stage + 1].append(dataset)
                    else:
                        # there is no x and y allocated
                        # by default, give it all the training data
                        number_unused_model = number_unused_model+1

                        # print("[WARN] The model {}-{} is not given any data".
                        #       format(stage + 1, next_model_id))
                        train_datas[stage + 1].append((None, None))
                print("unused model at stage {}: {}".format(
                    stage+1, number_unused_model))
        end_time = timer()

        y_pred = []
        for each in x_test:
            y_pred.append(self.predict(each))
        mse = metrics.mean_squared_error(y_test, y_pred)
        return mse, end_time - start_time

    def acceptable_next_model(self, raw_next_model_id, stage, isLeaf=False):
        if not isLeaf:
            stage=stage+1
        if raw_next_model_id <= 0:
            return 0
        elif raw_next_model_id >= self.num_of_models[stage]:
            return self.num_of_models[stage] - 1
        else:
            return raw_next_model_id

    def find_closed_prev_model_id(self, next_model_id, stage):
        while(self.models[stage][next_model_id] is None):
            next_model_id = next_model_id - 1
        return next_model_id

    def predict(self, key):
        next_model_id = 0
        final_output = 0
        for stage in range(self.num_of_stages):
            if not stage == self.num_of_stages - 1:
                output = self.models[stage][next_model_id].predict(key)
                next_model_id = int(output * self.num_of_models[stage + 1] /
                                    self.max_pos)
                next_model_id = self.acceptable_next_model(
                    next_model_id, stage)
            else:
                # leaf node reached
                # the output from the model is the predicted position directly
                next_model_id = self.acceptable_next_model(
                    next_model_id, stage, isLeaf=True)
                next_model_id = self.find_closed_prev_model_id(next_model_id,stage)
                final_output = self.models[stage][next_model_id].predict(key)
        return int(final_output)

    def get_staged_output(self, key, stage):
        if stage >= self.num_of_stages:
            raise ValueError("Stage cannot surpass the total number of stages")
        next_model_id = 0
        for stage in range(stage):
            output = self.models[stage][next_model_id].predict(key)
            next_model_id = int(output * self.num_of_models[stage + 1] /
                                self.max_pos)
            next_model_id = self.acceptable_next_model(next_model_id, stage)
        return next_model_id
