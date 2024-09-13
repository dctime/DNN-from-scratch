#include <SFML/Graphics.hpp>
#include <iostream>
#include <math.h>
#include <vector>

#define WINDOW_WIDTH 1920.0f
#define WINDOW_HEIGHT 1080.0f

#include <SFML/Graphics.hpp>
#include <iostream>

struct ImageRenderer {
  sf::RenderWindow &window;
  sf::Sprite sprite;
  sf::Vector2f topLeftPosition;
  sf::Vector2f size;

  sf::Vector2f getScreenLocation(int x, int y) const {
    return topLeftPosition +
           sf::Vector2f(x * (size.x / 28.0f), y * (size.y / 28.0f));
  }
};

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

void drawNeuralNetwork(sf::RenderWindow &window, const std::vector<int> &layers,
                       const std::vector<std::vector<float>> &weights,
                       const std::vector<float> &neuralValues, float offsetX,
                       float offsetY) {
  const float NEURON_RADIUS = 20.f;
  const float HORIZONTAL_SPACING = WINDOW_WIDTH / (layers.size() + 1);
  const sf::Color NEURON_COLOR = sf::Color::White;
  const sf::Color CONNECTION_COLOR = sf::Color::White;
  const sf::Color INACTIVE_COLOR =
      sf::Color(255, 255, 255, 50); // Faint color for inactive elements
  const float LINE_THICKNESS = 2.f;

  std::vector<std::vector<sf::CircleShape>> neurons;

  // Create and position neurons
  for (size_t i = 0; i < layers.size(); ++i) {
    std::vector<sf::CircleShape> layer_neurons;
    float x = (i + 1) * HORIZONTAL_SPACING + offsetX;
    float layer_height = WINDOW_HEIGHT / (layers[i] + 1);

    for (int j = 0; j < layers[i]; ++j) {
      sf::CircleShape neuron(NEURON_RADIUS);
      sf::Color neuronColor = NEURON_COLOR;
      neuronColor.a =
          static_cast<sf::Uint8>(neuralValues[i * layers[i] + j] *
                                 255); // Set alpha based on neuralValues
      neuron.setFillColor(neuronColor);
      neuron.setPosition(x - NEURON_RADIUS,
                         (j + 1) * layer_height - NEURON_RADIUS + offsetY);

      layer_neurons.push_back(neuron);
    }
    neurons.push_back(layer_neurons);
  }

  // Draw connections and neurons
  // window.clear(sf::Color::Black);

  // Draw connections
  for (size_t i = 0; i < neurons.size() - 1; ++i) {
    for (size_t j = 0; j < neurons[i].size(); ++j) {
      for (size_t k = 0; k < neurons[i + 1].size(); ++k) {
        sf::Vertex line[] = {
            sf::Vertex(neurons[i][j].getPosition() +
                           sf::Vector2f(NEURON_RADIUS, NEURON_RADIUS),
                       CONNECTION_COLOR),
            sf::Vertex(neurons[i + 1][k].getPosition() +
                           sf::Vector2f(NEURON_RADIUS, NEURON_RADIUS),
                       CONNECTION_COLOR)};
        line[0].color.a =
            static_cast<sf::Uint8>(weights[i][j * neurons[i + 1].size() + k] *
                                   255); // Set alpha based on weights
        line[1].color.a =
            static_cast<sf::Uint8>(weights[i][j * neurons[i + 1].size() + k] *
                                   255); // Set alpha based on weights
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

  ImageRenderer renderer{window, sf::Sprite(), sf::Vector2f(), sf::Vector2f()};

  // Example neural network structure: {3, 4, 4, 2}
  std::vector<int> layers = {3, 4, 4, 2};

  // Example weights and neural values
  std::vector<std::vector<float>> weights = {
      {0.5f, 0.8f, 0.3f, 0.9f, 0.4f, 0.7f, 0.2f, 0.6f, 0.1f, 0.5f, 0.8f, 0.3f},
      {0.6f, 0.7f, 0.2f, 0.5f, 0.9f, 0.4f, 0.8f, 0.3f, 0.7f, 0.2f, 0.6f, 0.1f},
      {0.4f, 0.5f, 0.8f, 0.3f, 0.7f, 0.2f, 0.6f, 0.1f}};
  std::vector<float> neuralValues = {0.9f, 0.8f, 0.7f, 0.6f, 0.5f, 0.4f, 0.3f,
                                     0.2f, 0.1f, 0.9f, 0.8f, 0.7f, 0.6f};

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

    renderImageInWindow(window, "../mnist_png/testing/0/3.png", position, size, renderer);
    // renderImageInWindow(window, "../mnist_png/testing/0/3.png", position, size);

    float offsetX = 100.f; // Example offset X
    float offsetY = 0.f;   // Example offset Y

    drawNeuralNetwork(window, layers, weights, neuralValues, offsetX, offsetY);

    window.display();
  }

  return 0;
}
