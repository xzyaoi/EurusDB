# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import csv
import os


def write_results(experiment_id, results):
    header = [
        "Name", "Build Time (s)", "Evaluation Time (s)",
        "Evaluation Error (MSE)", "Memory Size (KB)"
    ]
    with open(os.path.join("results", "{}.csv".format(experiment_id)),
              'w') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(header)
        for each in results:
            csvwriter.writerow(each)
