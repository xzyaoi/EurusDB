# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import csv
import os

from src.utilities.generator import get_data

DATA_SIZE = 10000
BLOCK_SIZE = 10


def generate_2d_data(distribution_x, distribution_y, data_size):
    X = get_data(distribution_x, data_size)
    Y = get_data(distribution_y, data_size)
    x_multiplicant = 1
    y_multiplicant = 1
    if distribution_x == "EXPONENTIAL":
        x_multiplicant = 100
    elif distribution_x == "LOGNORMAL":
        x_multiplicant = 10000
    if distribution_y == "EXPONENTIAL":
        y_multiplicant = 100
    elif distribution_y == "LOGNORMAL":
        y_multiplicant = 10000
    data_path = os.path.join(
        "data", "2d_"+distribution_x.lower()+"_"+ \
            distribution_y.lower()+"_"+str(data_size)+".csv"
    )
    with open(data_path, "w+") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["x", "y", "value"])
        for index, number in enumerate(X):
            csv_writer.writerow([
                int(number * x_multiplicant),
                int(Y[index] * y_multiplicant), index // BLOCK_SIZE
            ])


if __name__ == "__main__":
    x_distribution = sys.argv[1]
    y_distribution = sys.argv[2]
    data_size = int(sys.argv[3])
    generate_2d_data(x_distribution, y_distribution, data_size)
