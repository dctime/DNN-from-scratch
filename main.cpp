#include <SFML/Graphics.hpp>
#include <iostream>
#include <math.h>
#include <random>
#include <vector>

#define WINDOW_WIDTH 1920.0f
#define WINDOW_HEIGHT 720.0f

#include <SFML/Graphics.hpp>
#include <iostream>

struct ImageRenderer {
  sf::RenderWindow &window;
  sf::Sprite sprite;
  sf::Vector2f topLeftPosition;
  sf::Vector2f size;

  sf::Vector2f getScreenLocation(int x, int y) const {
    if (x > 27)
      x = 27;
    if (y > 27)
      y = 27;

    sf::Vector2f pixelSize(size.x / 28.0f, size.y / 28.0f);
    return topLeftPosition + sf::Vector2f(x * pixelSize.x + pixelSize.x / 2.0f,
                                          y * pixelSize.y + pixelSize.y / 2.0f);
  }

  void renderPoints() {
    for (int x = 0; x < 28; ++x) {
      for (int y = 0; y < 28; ++y) {
        sf::Vector2f position = getScreenLocation(x, y);
        sf::CircleShape point(2); // Small circle to represent the point
        point.setPosition(position);
        point.setFillColor(sf::Color::Red);
        window.draw(point);
      }
    }
  }
};

struct NeuralNetwork {
  std::vector<int> layers;
  std::vector<std::vector<float>> weights;
  std::vector<float> neuralValues;
};

void createNeuralNetwork(NeuralNetwork &nn) {
  nn.layers = {784, 64, 64,
               10}; // Example layers: 784 input nodes, 128 in the first hidden
                    // layer, 64 in the second hidden layer, and 10 output nodes

  // Initialize weights with random values
  srand(static_cast<unsigned>(time(0))); // Seed for random number generation
  for (size_t i = 0; i < nn.layers.size() - 1; ++i) {
    std::vector<float> layer_weights;
    for (int j = 0; j < nn.layers[i] * nn.layers[i + 1]; ++j) {
      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_real_distribution<float> dist(-1.0, 1.0);
      layer_weights.push_back(
          static_cast<float>(dist(gen))); // Random weights between 0 and 1
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
}

void renderImageInWindow(sf::RenderWindow &window, const std::string &imagePath,
                         sf::Vector2f position, sf::Vector2f size,
                         ImageRenderer &renderer) {
  // Load the texture
  sf::Texture texture;
  if (!texture.loadFromFile(imagePath)) {
    std::cerr << "Error loading image" << std::endl;
    return;
  }

  // Create a sprite
  sf::Sprite sprite;
  sprite.setTexture(texture);

  // Set the size
  sprite.setScale(size.x / texture.getSize().x, size.y / texture.getSize().y);

  // Calculate the top-left position
  sf::Vector2f topLeftPosition =
      position - sf::Vector2f(size.x / 2.0f, size.y / 2.0f);

  // Set the position
  sprite.setPosition(topLeftPosition);

  // Draw the sprite
  window.draw(sprite);

  // Draw white lines to cut the image into 28x28 pixels
  sf::RectangleShape line(sf::Vector2f(size.x, 1)); // Horizontal line
  line.setFillColor(sf::Color(128, 128, 128));

  for (int i = 1; i < 28; ++i) {
    line.setPosition(topLeftPosition.x,
                     topLeftPosition.y + i * (size.y / 28.0f));
    window.draw(line);
  }

  line.setSize(sf::Vector2f(1, size.y)); // Vertical line

  for (int i = 1; i < 28; ++i) {
    line.setPosition(topLeftPosition.x + i * (size.x / 28.0f),
                     topLeftPosition.y);
    window.draw(line);
  }

  // Draw borders
  sf::RectangleShape border(sf::Vector2f(size.x, 1)); // Top border
  border.setFillColor(sf::Color::White);
  border.setPosition(topLeftPosition.x, topLeftPosition.y);
  window.draw(border);

  border.setSize(sf::Vector2f(size.x, 1)); // Bottom border
  border.setPosition(topLeftPosition.x, topLeftPosition.y + size.y);
  window.draw(border);

  border.setSize(sf::Vector2f(1, size.y)); // Left border
  border.setPosition(topLeftPosition.x, topLeftPosition.y);
  window.draw(border);

  border.setSize(sf::Vector2f(1, size.y)); // Right border
  border.setPosition(topLeftPosition.x + size.x, topLeftPosition.y);
  window.draw(border);

  // Update the renderer struct
  renderer.sprite = sprite;
  renderer.topLeftPosition = topLeftPosition;
  renderer.size = size;
}

void drawNeuralNetwork(sf::RenderWindow &window, const NeuralNetwork &nn,
                       float offsetX, float offsetY) {
  const float NEURON_RADIUS = 3.f; // Smaller radius
  const float HORIZONTAL_SPACING =
      WINDOW_WIDTH / (nn.layers.size() + 2); // Increase spacing
  const sf::Color NEURON_COLOR = sf::Color::White;
  const sf::Color INACTIVE_COLOR =
      sf::Color(255, 255, 255, 50); // Faint color for inactive elements
  const float LINE_THICKNESS = 2.f;
  const int MAX_COUNT_PER_COLUMN = 28;

  std::vector<std::vector<sf::CircleShape>> neurons;

  // Create and position neurons
  for (size_t i = 0; i < nn.layers.size(); ++i) {
    std::vector<sf::CircleShape> layer_neurons;

    for (int j = 0; j < nn.layers[i]; ++j) {
      sf::CircleShape neuron(NEURON_RADIUS);
      sf::Color neuronColor = NEURON_COLOR;
      neuronColor.a =
          static_cast<sf::Uint8>(nn.neuralValues[i * nn.layers[i] + j] *
                                 255); // Set alpha based on neuralValues
      neuron.setFillColor(neuronColor);
      float x = (i + 1) * HORIZONTAL_SPACING + offsetX - NEURON_RADIUS +
                (NEURON_RADIUS * 3) * (int)(j / MAX_COUNT_PER_COLUMN);
      // float layer_height = WINDOW_HEIGHT / (((j > (nn.layers[i] -
      // MAX_COUNT_PER_COLUMN)) ? (nn.layers[i] % MAX_COUNT_PER_COLUMN) :
      // (MAX_COUNT_PER_COLUMN)) + 2); // Increase spacing

      int nodesPerRow =
          ((j / MAX_COUNT_PER_COLUMN + 1) * MAX_COUNT_PER_COLUMN) > nn.layers[i]
              ? nn.layers[i] % MAX_COUNT_PER_COLUMN
              : MAX_COUNT_PER_COLUMN;
      float layer_height = WINDOW_HEIGHT / ((nodesPerRow) + 2);
      float y = ((j % MAX_COUNT_PER_COLUMN) + 1) * layer_height -
                NEURON_RADIUS + offsetY;
      neuron.setPosition(x, y);

      layer_neurons.push_back(neuron);
    }
    neurons.push_back(layer_neurons);
  }

  // Draw connections and neurons

  // Draw connections
  for (size_t i = 0; i < neurons.size() - 1; ++i) {
    for (size_t j = 0; j < neurons[i].size(); ++j) {
      for (size_t k = 0; k < neurons[i + 1].size(); ++k) {
        float weight = nn.weights[i][j * neurons[i + 1].size() + k];
        sf::Color lineColor;
        if (weight >= 0) {
          lineColor = sf::Color(
              255, 0, 0,
              static_cast<sf::Uint8>(weight * 64)); // Red for positive weights, MAX set to 128
        } else {
          lineColor =
              sf::Color(0, 255, 0,
                        static_cast<sf::Uint8>(
                            -weight * 64)); // Green for negative weights, MAX set to 128
        }
        sf::Vertex line[] = {
            sf::Vertex(neurons[i][j].getPosition() +
                           sf::Vector2f(NEURON_RADIUS, NEURON_RADIUS),
                       lineColor),
            sf::Vertex(neurons[i + 1][k].getPosition() +
                           sf::Vector2f(NEURON_RADIUS, NEURON_RADIUS),
                       lineColor)};
        if (line[0].color.a == 0) {
          line[0].color = INACTIVE_COLOR;
          line[1].color = INACTIVE_COLOR;
        }
        window.draw(line, 2, sf::Lines);
      }
    }
  }

  // Draw neurons
  for (const auto &layer : neurons) {
    for (const auto &neuron : layer) {
      window.draw(neuron);
    }
  }
}

int main() {
  sf::RenderWindow window(sf::VideoMode(WINDOW_WIDTH, WINDOW_HEIGHT),
                          "Neural Network Visualization");

  NeuralNetwork nn;
  createNeuralNetwork(nn);

  // Print the size of each layer to verify
  for (size_t i = 0; i < nn.layers.size(); ++i) {
    std::cout << "Layer " << i << ": " << nn.layers[i] << " neurons"
              << std::endl;
  }

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

    ImageRenderer renderer{window, sf::Sprite(), sf::Vector2f(),
                           sf::Vector2f()};

    int firstLayerSize;
    std::vector<sf::Vector2f> firstLayerNodes;

    renderImageInWindow(window, "../mnist_png/testing/0/3.png", position, size,
                        renderer);
    // renderImageInWindow(window, "../mnist_png/testing/0/3.png", position,
    // size);

    float offsetX = 100.f; // Example offset X
    float offsetY = 0.f;   // Example offset Y

    renderer.renderPoints();
    // Draw the neural network
    drawNeuralNetwork(window, nn, offsetX, offsetY);

    window.display();
  }

  return 0;
}
