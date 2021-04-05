# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import sys
import uuid
from typing import List

import numpy as np
import pandas as pd
from tabulate import tabulate

from src.indexing.models import BaseModel
from src.indexing.models.ml.polynomial_regression import PRModel
from src.indexing.models.nn.fcn import FCNModel
from src.indexing.models.rmi.staged import StagedModel
from src.indexing.models.trees.b_tree import BTreeModel
from src.indexing.utilities.metrics import get_memory_size
from src.indexing.utilities.results import write_results
from src.queries.point import PointQuery

ratio = 0.2
b_tree_degree = 20
experiment_id = uuid.uuid4().__str__()[0:8]


def load_1D_Data(filename):
    data = pd.read_csv(filename)
    test_data = data.sample(n=int(ratio * len(data)))
    page_size = len(data) // np.max(data.iloc[:, 1])
    return data, test_data, page_size


def evaluate(filename):
    data, test_data, page_size = load_1D_Data(filename)
    btm = BTreeModel(page_size, b_tree_degree)
    fcnm = FCNModel(page_size=page_size,
                    layers=[1, 32, 32, 1],
                    activations=['relu', 'relu', 'relu'])

    lrm = PRModel(1, page_size)
    prm = PRModel(2, page_size)
    sgm1 = StagedModel(['fcn', 'fcn', 'fcn'], [1, 200, 4000], page_size)
    models = [sgm1, btm, fcnm, lrm, prm]
    ptq = PointQuery(models)
    build_times = ptq.build(data, ratio)
    mses, eval_times = ptq.evaluate(test_data)
    result = []
    header = [
        "Name", "Build Time (s)", "Evaluation Time (s)",
        "Evaluation Error (MSE)", "Memory Size (KB)"
    ]
    for index, model in enumerate(models):
        result.append([
            model.name, build_times[index], eval_times[index], mses[index],
            get_memory_size(model)
        ])
    write_results(experiment_id, result)
    print("Experiment ID: {}".format(experiment_id))
    print(tabulate(result, header))
    models_predict(data, models)


def models_predict(data, models: List[BaseModel]):
    x = data.iloc[:, :-1].values
    x = x.reshape(-1)
    gt_y = data.iloc[:, 1].values
    pred_ys = []
    for model in models:
        pred_y = []
        for each in x:
            pred_y.append(int(model.predict(each) // model.page_size))
        if model.name == 'Fully Connected Neural Network':
            # print(pred_y)
            pass
        pred_ys.append(pred_y)
    results = {}
    results['x'] = x
    results['ground_truth'] = gt_y
    for idx, model in enumerate(models):
        results[model.name] = pred_ys[idx]
    df = pd.DataFrame.from_dict(results)
    df.to_csv('result_10k_{}.csv'.format(experiment_id), index=False)
    print("Results of experiment {} have been saved to result.csv".format(
        experiment_id))


if __name__ == "__main__":
    filename = sys.argv[1]
    evaluate(filename)
