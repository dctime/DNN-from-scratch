#include <ctime>
#include <iostream>
#include <random>
#include <stdexcept>
#include <vector>

#include "neural_network.h"

// Function to change the values of the first layer
void NeuralNetwork::changeFirstLayerValues(Eigen::MatrixXd& newValues) {
  if (layers.empty() || layers[0] <= 0) {
    throw std::runtime_error("First layer is not properly defined.");
  }

  // Ensure the new values match the size of the first layer
  if (newValues.rows() != layers[0]) {
    throw std::invalid_argument("Size of new values does not match the size of the first layer.");
  }

  // Update the neural values for the first layer
  neuralValues[0] = newValues;
}

void createNeuralNetwork(NeuralNetwork &nn, const std::vector<int>& layersNodeCount) {
  nn.layers = layersNodeCount;

  // Clear existing values
  nn.weights.clear();
  nn.neuralValues.clear();
  nn.bias.clear();

  // Seed for random number generation
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<float> weight_dist(-1.0, 1.0);
  std::uniform_real_distribution<float> bias_dist(-1.0, 1.0);
  std::uniform_real_distribution<float> neural_value_dist(0.0, 1.0);

  // Initialize weights with random values
  for (size_t i = 0; i < nn.layers.size() - 1; ++i) {
    Eigen::MatrixXd layer_weights = Eigen::MatrixXd::Zero(nn.layers[i+1], nn.layers[i]);
    for (int row = 0; row < layer_weights.rows(); ++row) {
      for (int col = 0; col < layer_weights.cols(); ++col) {
        layer_weights(row, col) = weight_dist(gen);
      }
    }
    nn.weights.push_back(layer_weights);
  }

  // Initialize neural values with random values
  for (int layer_size : nn.layers) {
    Eigen::MatrixXd layer_values = Eigen::MatrixXd::Zero(layer_size, 1);
    for (int row = 0; row < layer_values.rows(); ++row) {
      layer_values(row, 0) = neural_value_dist(gen);
    }
    nn.neuralValues.push_back(layer_values);
  }

  // Initialize biases with random values
  for (size_t i = 1; i < nn.layers.size(); ++i) {
    Eigen::MatrixXd layer_biases = Eigen::MatrixXd::Zero(nn.layers[i], 1);
    for (int row = 0; row < layer_biases.rows(); ++row) {
      layer_biases(row, 0) = bias_dist(gen);
    }
    nn.bias.push_back(layer_biases);
  }
}

void NeuralNetwork::forwardPropagation(int layer) {
  if (layer <= 0 || layer >= layers.size()) {
    throw std::out_of_range("Layer index out of range or invalid.");
  }

  // std::cout << "Weights Size: " << weights[layer-1].rows() << "x" << weights[layer-1].cols();
  // std::cout << "Neural Values Size: " << neuralValues[layer-1].rows() << "x" << neuralValues[layer-1].cols();
  // std::cout << "Bias Size: " << bias[layer-1].rows() << "x" << bias[layer-1].cols();

  neuralValues[layer] = weights[layer-1] * neuralValues[layer-1] + bias[layer-1];
}
