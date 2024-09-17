#include <SFML/Graphics.hpp>
#include <iostream>
#include <math.h>
#include <vector>

#include "constants.h"

#include <SFML/Graphics.hpp>
#include <iostream>

#include "neural_network.cpp"
#include "render.cpp"

int main() {
  sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                          "Neural Network Visualization");
  // Neural Network Init
  NeuralNetwork nn;
  std::vector<int> layers = {28 * 28, 8, 8, 3};
  createNeuralNetwork(nn, layers);

  // Print the size of each layer to verify
  for (size_t i = 0; i < nn.layers.size(); ++i) {
    std::cout << "Layer " << i << ": " << nn.layers[i] << " neurons"
              << std::endl;
  }

  // Image renderer Init
  ImageRenderer renderer{window, sf::Sprite(), sf::Vector2f(), sf::Vector2f()};

  while (window.isOpen()) {
    sf::Event event;
    while (window.pollEvent(event)) {
      if (event.type == sf::Event::Closed)
        window.close();
    }

    window.clear();

    // Render the image
    sf::Vector2f position(
        200,
        window.getSize().y / 2.0f); // Position where the image will be rendered
    sf::Vector2f size(300, 300);    // Size of the image

    int firstLayerSize;
    std::vector<sf::Vector2f> firstLayerNodes;

    renderImageInWindow(window, "../mnist_png/testing/0/3.png", position, size,
                        renderer);

    // New values for the first layer
    std::vector<float> newValues;
    for (int x = 0; x < 28; x++) {
      for (int y = 0; y < 28; y++) {
        newValues.push_back(renderer.getGrayscaleValue(x, y));
      }
    }

    // Change the values of the first layer
    try {
      nn.changeFirstLayerValues(newValues);

    } catch (const std::exception &e) {
      std::cerr << "Error: " << e.what() << std::endl;
    }

    // Draw the neural network
    float offsetX = 100.f; // Example offset X
    float offsetY = 0.f;   // Example offset Y
    drawNeuralNetwork(window, nn, offsetX, offsetY);

    window.display();
  }

  return 0;
}
