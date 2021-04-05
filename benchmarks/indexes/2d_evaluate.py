# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
'''
// In the end, every function needs to have a strucutr like this. 
def _switch_to_np_array(input_):
        r"""
        Check the input, if it's not a Numpy array transform it to one.
        Parameters
        ----------
        input_ : array_like
            The object that requires a check.
        Returns
        -------
        input_ : ndarray
            The input data that's been transformed when required.
        """
        if isinstance(input_, np.ndarray) is False:
            input_ = np.array(input_)
        return input_
'''
from typing import List

import numpy as np
import pandas as pd
from tabulate import tabulate

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.indexing.models.lisa.basemodel import LisaBaseModel
from src.indexing.models.lisa.lisa import LisaModel
from src.indexing.models.trees.KD_tree import KDTreeModel
from src.indexing.models.trees.scipykdtree import ScipyKDTreeModel
from src.queries.point import PointQuery
from src.queries.range import RangeQuery

ratio = 0.5


def load_2D_Data(filename):
    data = pd.read_csv(filename)
    #data = data[0:100]
    # Remove duplicates
    col_names = list(data.columns)[:-1]
    data.drop_duplicates(subset=col_names, ignore_index=True, inplace=True)
    data = data[0:10000]
    test_data = data  #data.sample(n=int(ratio * len(data)))
    return data, test_data


def create_models(filename):
    data, test_data = load_2D_Data(filename)
    LisaBaseModel(100)
    KDTreeModel()
    scipykdtree = ScipyKDTreeModel(leafsize=10)
    lisa = LisaModel(cellSize=4, nuOfShards=5)

    #models = [lisaBm, kdtree, scipykdtree, lisa]
    models = [lisa, scipykdtree]
    #models = [scipykdtree]
    ptq = PointQuery(models)
    build_times = ptq.build(data, 0.002, use_index=False)
    return (models, ptq, test_data, build_times)
    # print("Build time",build_times)


def point_query_eval(models, ptq, test_data, build_times):
    i = 10
    result = []
    header = [
        "Name", "Test Data Size", "Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)", "Evaluation Error (MSE)"
    ]
    while (i <= 10000):
        #Sample a point
        mses, eval_times = ptq.evaluate_point(test_data.iloc[:i, :])

        for index, model in enumerate(models):
            result.append([
                model.name, i, build_times[index], eval_times[index],
                eval_times[index] / i, mses[index]
            ])
        print(len(result))
        i = i * 10
    print(tabulate(result, header))


def range_query_eval(models, ptq, test_data, build_times):
    i = 100
    result = []
    header = [
        "Name", "Query Size", "Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)", "Evaluation Error (MSE)"
    ]
    print(test_data.size)
    print(test_data.shape)
    idx = np.random.randint(test_data.shape[0] - 1001)
    while (i <= 4000):
        print(idx)
        query_l = test_data.iloc[idx, 0:2]
        if (idx + i) > test_data.shape[0]:
            break

        query_h = test_data.iloc[idx + i, 0:2]
        print('idx = %d i = %d ' % (idx, idx + i))
        print('query_l = %d %d' % (query_l[0], query_l[1]))
        print('query_h = %d %d' % (query_h[0], query_h[1]))
        test_range_query = test_data.iloc[idx:idx + i, :]
        mses, eval_times = ptq.evaluate_range_query(test_range_query)
        for index, model in enumerate(models):
            result.append([
                model.name, i, build_times[index], eval_times[index],
                eval_times[index] / i, mses[index]
            ])

        print(len(result))

        i = i + 100
    print(tabulate(result, header))


def knn_query_eval(models, ptq, test_data, build_times):
    i = 3
    result = []
    header = [
        "Name", "K Value", "Build Time (s)", "Evaluation Time (s)",
        "Average Evaluation Time (s)", "Evaluation Error (MSE)"
    ]
    print(test_data.size)
    print(test_data.shape)
    idx = np.random.randint(test_data.shape[0] - 1001)
    while (i <= 20):
        print(idx)
        query = test_data.iloc[idx, 0:2]
        if (idx + i) > test_data.shape[0]:
            break

        print('idx = %d i = %d ' % (idx, idx + i))
        print('query_l = %d %d' % (query[0], query[1]))

        y_gt = ptq.evaluate_scipy_kdtree_knn_query(query, k=i)
        mses, eval_times = ptq.evaluate_knn_query(query, y_gt, k=i)

        for index, model in enumerate(models):
            if (model.name == 'Scipy KD-Tree') or (model.name
                                                   == 'Lisa Baseline'):
                continue
            result.append([
                model.name, i, build_times[index], eval_times[index],
                eval_times[index] / i, mses[index]
            ])

        print(len(result))

        i = i + 1
    print(tabulate(result, header))


'''
def evaluate_range(filename):
    data, test_data = load_2D_Data(filename)
    kdtree = KDTreeModel()
    lisa = LisaModel(cellSize=10, nuOfShards=5)

    # models = [lisaBm, kdtree, scipykdtree, lisa]
    models = [lisa, kdtree]       #add trees here
    rq = RangeQuery(models)
    build_times = rq.build(data, 0.00002)

    lower_left = []
    upper_right = []
    for i in range(2):
        lower_left.append(randrange(100,200))
        upper_right.append(randrange(200,300))

    
    # Area = (x_min,y_min,x_max,y_max)
    area=(lower_left[0],lower_left[1],upper_right[0],upper_right[1])
    
    range_points=rq.range_query(area)

    print(range_points)

    return range_points
'''


def models_predict_point(data, models: List[BaseModel]):
    data = data.to_numpy()
    x = data[:, :-1]
    gt_y = data[:, -1:].reshape(-1)
    pred_ys = []
    for model in models:
        pred_y = []
        for each in x:
            pred_y.append(int(model.predict(each)))
        pred_ys.append(pred_y)
    results = {}
    results['x1'] = x[:, 0]
    results['x2'] = x[:, 1]
    results['ground_truth'] = gt_y

    for idx, model in enumerate(models):
        results[model.name] = pred_ys[idx]
        print('mse error for model %s is %f' %
              (model.name,
               metrics.mean_squared_error(np.array(pred_ys[idx]), gt_y)))

    df = pd.DataFrame.from_dict(results)
    df.to_csv('result.csv', index=False)
    print("Results have been saved to result.csv")


if __name__ == "__main__":
    # filename = sys.argv[1]
    filename = 'data/2d_lognormal_lognormal_1000000.csv'

    # evaluate(filename)
    (models, ptq, test_data, build_times) = create_models(filename)
    #range_query_eval(models, ptq, test_data,build_times)
    knn_query_eval(models, ptq, test_data, build_times)
