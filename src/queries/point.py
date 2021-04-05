# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

from timeit import default_timer as timer
from typing import List

import numpy as np

import src.indexing.utilities.metrics as metrics
from src.indexing.models import BaseModel
from src.queries import Query


class PointQuery(Query):
    def __init__(self, models: List[BaseModel]) -> None:
        super().__init__(models)

    def predict(self, model_idx: int, key: int):
        return self.models[model_idx].predict(key)

    def predict_range_query(self, model_idx: int, query_l, query_u):
        print('Get keys in range (%d, %d), (%d, %d)' %
              (query_l[0], query_l[1], query_u[0], query_u[1]))
        return self.models[model_idx].predict_range_query(query_l, query_u)

    def predict_knn_query(self, model_idx: int, query, k):

        return self.models[model_idx].predict_knn_query(query, k)

    def evaluate_point(self, test_data):
        data_size = test_data.shape[0]
        print("[Point Query] Evaluating {} datapoints".format(data_size))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            ys = []
            start_time = timer()
            for i in range(data_size):
                y = self.predict(idx, test_data.iloc[i, :-1])
                y = int(y // model.page_size)
                ys.append(y)
            end_time = timer()
            yhat = np.array(ys).reshape(-1, 1)
            ytrue = np.array(test_data.iloc[:, -1:])
            mse = metrics.mean_squared_error(yhat, ytrue)
            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)
        return mses, build_times

    def evaluate_range_query(self, test_range_query):
        data_size = test_range_query.shape[0]
        print("[Point Query] Evaluating {} datapoints".format(data_size))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            if (model.name == 'Lisa Baseline'):
                continue
            start_time = timer()
            y_pred = self.predict_range_query(idx,
                                              test_range_query.iloc[0, :-1],
                                              test_range_query.iloc[-1, :-1])
            end_time = timer()
            if (y_pred.shape[0] != data_size):
                print(
                    'Nu of predicted entries in range query %d versus expected entries %d',
                    y_pred.shape[0], data_size)
                mse = -1
                exit(0)
            else:
                yhat = np.array(y_pred).reshape(-1, 1)
                ytrue = np.array(test_range_query.iloc[:, -1:])
                mse = metrics.mean_squared_error(yhat, ytrue)
                '''
                if(mse != 0):
                    print(yhat)
                    print('\n\n\n\n')
                    print(ytrue)
                    for i in range (data_size):
                        if(yhat[i] != ytrue[i]):
                            print( ' Predicted y %d Expected y %d' %(yhat[i], ytrue[i]))
                '''

            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)

        return mses, build_times

    def evaluate_scipy_kdtree_knn_query(self, query, k):

        print("Get %d nearest neighbours for query %d %d" %
              (k, query[0], query[1]))
        for idx, model in enumerate(self.models):
            if model.name == 'Scipy KD-Tree':
                y_pred = self.models[idx].predict(query, k)
                print('Grounftruth for query %d %d for %d neighbours' %
                      (query[0], query[1], k))
                print(y_pred)
                return y_pred
            else:
                continue
        return -1

    def evaluate_knn_query(self, query, ytrue, k):

        print("[Point Query %d %d]  Evaluating %d neighbours" %
              (query[0], query[1], k))
        build_times = []
        mses = []
        for idx, model in enumerate(self.models):
            if (model.name == 'Scipy KD-Tree') or (model.name
                                                   == 'Lisa Baseline'):
                continue
            start_time = timer()
            y_pred = self.predict_knn_query(idx, query, k)
            end_time = timer()
            if (y_pred.shape[0] != ytrue.shape[0]):
                print(
                    'Nu of predicted entries in range query %d versus expected entries %d',
                    y_pred.shape[0], ytrue.shape[0])
                mse = -1
                exit(0)
            else:
                yhat = np.array(y_pred).reshape(-1, 1)
                mse = metrics.mean_squared_error(yhat, ytrue)

                if (mse != 0):
                    print(yhat)
                    print('\n\n\n\n')
                    print(ytrue)
                    for i in range(ytrue.shape[0]):
                        if (yhat[i] != ytrue[i]):
                            print(' Predicted y %d Expected y %d' %
                                  (yhat[i], ytrue[i]))

            mses.append(mse)
            print("{} model tested in {:.4f} seconds with mse {:.4f}".format(
                model.name, end_time - start_time, mse))
            build_times.append(end_time - start_time)

        return mses, build_times

    def get_model(self, model_idx: int):
        return self.models[model_idx]
