#pragma once

#include <vector>
#include <Eigen/Dense>

struct NeuralNetwork {
  std::vector<int> layers;
  std::vector<Eigen::MatrixXd> weights;
  std::vector<Eigen::MatrixXd> neuralValues;
  std::vector<Eigen::MatrixXd> bias; // Added bias member
  
  void changeFirstLayerValues(Eigen::MatrixXd& newValues);

  // layer much be greater than 0
  void forwardPropagation(int layer);
};

void createNeuralNetwork(NeuralNetwork &nn, const std::vector<int>& layersNodeCount);
