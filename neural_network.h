#pragma once

#include <vector>

struct NeuralNetwork {
  std::vector<int> layers;
  std::vector<std::vector<float>> weights;
  std::vector<float> neuralValues;
  std::vector<float> bias; // Added bias member
  
  void changeFirstLayerValues(const std::vector<float> &newValues);
  void setBiases(const std::vector<float> &newBiases);
};

