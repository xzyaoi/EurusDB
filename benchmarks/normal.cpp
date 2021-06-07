// Copyright (c) 2021 Xiaozhe Yao et al.
//
// This software is released under the MIT License.
// https://opensource.org/licenses/MIT

#include <algorithm>
#include <climits>
#include <fstream>
#include <iostream>
#include <random>
#include <set>
#include <vector>

int main(int argc, char **argv)
{
  int nelements = 190000000;
  if (argc > 2)
  {
    std::cerr << "Wrong Number of Arguments" << std::endl;
  }
  nelements = atoi(argv[1]);
  std::cout << "Generating "<<nelements<< " elements..." << std::endl;
  double scale = 1e+11;
  double max = double(INT_MAX) / scale;
  int block_size = 10;
  std::mt19937 rng;
  rng.seed(std::random_device()());
  std::normal_distribution<double> dist(0.0, 1.0);

  std::set<int> samples;

  while (samples.size() < nelements)
  {
    double r = dist(rng);
    if (r > max)
      continue;
    samples.insert(int(r * scale));
    if (samples.size() % 1000000 == 0)
    {
      std::cerr << "Generated " << samples.size() << std::endl;
    }
  }

  std::vector<int> vec(samples.begin(), samples.end());
  std::sort(vec.begin(), vec.end());
  std::cerr << "min = " << vec[0] << std::endl;
  std::cerr << "max = " << vec[vec.size() - 1] << std::endl;

  std::ofstream myfile;
  myfile.open("data/1d_normal_" + std::to_string(nelements) + ".csv");
  myfile << "val,block\n";
  for (std::vector<int>::size_type i = 1; i != vec.size(); i++)
  {
    myfile << std::to_string(vec[i] - vec[0] + 1) + "," +
                  std::to_string(i / block_size) + "\n";
  }
  myfile.close();
  return 0;
}