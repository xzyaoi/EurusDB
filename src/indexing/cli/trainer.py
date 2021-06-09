# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT
import pandas as pd

from src.indexing.models.ml.polynomial_regression import PRModel
from src.indexing.models.ml.rdp_polylines import RDPModel
from src.indexing.utilities.trainer import ModelTrainer


def train(args):
    models = []
    if 'polynomial' in args:
        model = PRModel(args['polynomial']['degree'])
        model.savepath = args['polynomial']['savepath']
        models.append(model)
    if 'rdp' in args:
        model = RDPModel(args['rdp']['epsilon'])
        model.savepath = args['rdp']['savepath']
        models.append(model)
    trainer = ModelTrainer(models)
    data = pd.read_csv(args['filepath'])
    mses, build_times = trainer.build(data, args['test_ratio'],
                                      args['use_index'], args['sample_ratio'])
    models = trainer.get_models()
    for each in models:
        print(each.export())
        each.save()
    print("Finished")


if __name__ == "__main__":
    args = {
        "polynomial": {
            "degree": 1,
            "savepath": "models/polynomial.json"
        },
        "rdp": {
            "epsilon": 0.01,
            "savepath": "models/rdp.json"
        },
        "filepath": "data/1d_lognormal_1000000.csv",
        "test_ratio": 0.3,
        "use_index": True,
        "sample_ratio": 1
    }
    train(args)
