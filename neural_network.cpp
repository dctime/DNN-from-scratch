#include <ctime>
#include <random>
#include <stdexcept>
#include <vector>

#include "neural_network.h"

// Function to change the values of the first layer
void NeuralNetwork::changeFirstLayerValues(const std::vector<float> &newValues) {
  if (!layers.empty() && layers[0] > 0) {
    // Ensure the new values match the size of the first layer
    if (newValues.size() == layers[0]) {
      for (std::size_t i = 0; i < newValues.size(); ++i) {
        neuralValues[i] = newValues[i];
      }
    } else {
      // Handle the case where the size does not match
      throw std::invalid_argument(
          "Size of new values does not match the size of the first layer.");
    }
  } else {
    throw std::runtime_error("First layer is not properly defined.");
  }
}

// Function to set the bias values
void NeuralNetwork::setBiases(const std::vector<float> &newBiases) {
  if (newBiases.size() == bias.size()) {
    for (size_t i = 0; i < newBiases.size(); ++i) {
      bias[i] = newBiases[i];
    }
  } else {
    // Handle the case where the size does not match
    throw std::invalid_argument(
        "Size of new biases does not match the size of the bias vector.");
  }
}

void createNeuralNetwork(NeuralNetwork &nn, std::vector<int> layersNodeCount) {
  nn.layers = layersNodeCount;

  // Clear existing values
  nn.weights.clear();
  nn.neuralValues.clear();
  nn.bias.clear();

  // Initialize weights with random values
  srand(static_cast<unsigned>(time(0))); // Seed for random number generation
  for (size_t i = 0; i < nn.layers.size() - 1; ++i) {
    std::vector<float> layer_weights;
    for (int j = 0; j < nn.layers[i] * nn.layers[i + 1]; ++j) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(-1.0, 1.0);
      layer_weights.push_back(
          static_cast<float>(dist(gen))); // Random weights between -1 and 1
    }
    nn.weights.push_back(layer_weights);
  }

  // Initialize neural values with random values
  for (int layer_size : nn.layers) {
    for (int j = 0; j < layer_size; ++j) {
      nn.neuralValues.push_back(
          static_cast<float>(rand()) /
          RAND_MAX); // Random neural values between 0 and 1
    }
  }

  // Initialize biases with random values
  for (size_t i = 1; i < nn.layers.size(); ++i) {
    std::vector<float> layer_biases(nn.layers[i]);
    for (int j = 0; j < nn.layers[i]; ++j) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(-1.0, 1.0);
      nn.bias.push_back(
          static_cast<float>(dist(gen))); // Random biases between -1 and 1
    }
  }
}
