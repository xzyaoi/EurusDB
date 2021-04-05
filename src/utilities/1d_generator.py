# Copyright (c) 2021 Xiaozhe Yao et al.
#
# This software is released under the MIT License.
# https://opensource.org/licenses/MIT

import csv
import os

from src.utilities.generator import get_data

DATA_SIZE = 10000
BLOCK_SIZE = 10
FACTOR = 10


def generate_1d_data(distribution, data_size=DATA_SIZE):
    data = get_data(distribution, data_size)
    multiplicant = 1
    if distribution == "EXPONENTIAL":
        multiplicant = 100
    elif distribution == "LOGNORMAL":
        multiplicant = 10000
    data_path = os.path.join(
        "data", "1d_" + distribution.lower() + "_" + str(data_size) + ".csv")
    with open(data_path, "w+") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["val", "block"])
        for index, number in enumerate(data):
            csv_writer.writerow(
                [int(number * multiplicant), index // BLOCK_SIZE])


if __name__ == "__main__":
    distribution = sys.argv[1]
    data_size = int(sys.argv[2])
    generate_1d_data(distribution.upper(), data_size)
