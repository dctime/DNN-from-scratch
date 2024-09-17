#include "constants.h"

#include <SFML/Graphics.hpp>
#include <iostream>

#include "neural_network.h"
#include "render.h"

// use renderImageInWindow() to update the renderer

sf::Vector2f ImageRenderer::getScreenLocation(int x, int y) const {
  if (x > 27)
    x = 27;
  if (y > 27)
    y = 27;

  sf::Vector2f pixelSize(size.x / 28.0f, size.y / 28.0f);
  return topLeftPosition + sf::Vector2f(x * pixelSize.x + pixelSize.x / 2.0f,
                                        y * pixelSize.y + pixelSize.y / 2.0f);
}

// New function to get grayscale value at (x, y)
float ImageRenderer::getGrayscaleValue(int x, int y) const {
  if (x < 0 || x >= 28 || y < 0 || y >= 28) {
    throw std::out_of_range("Coordinates out of range");
  }

  sf::Color color = image.getPixel(x, y);

  // Since the image is grayscale, the red, green, and blue values are the
  // same std::cout << "R: " << static_cast<int>(color.r) << std::endl;
  return (color.r / 255.0);
}

// New function to update the image
void ImageRenderer::updateImage() {
  image = sprite.getTexture()->copyToImage();
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

  renderer.updateImage();
}

static std::vector<std::vector<sf::CircleShape>>
createNeuronsForRendering(const NeuralNetwork &nn, float offsetX,
                          float offsetY) {
  const float NEURON_RADIUS = 3.f;
  const float HORIZONTAL_SPACING = WINDOW_WIDTH / (nn.layers.size() + 2);
  const sf::Color NEURON_COLOR = sf::Color::White;
  const int MAX_COUNT_PER_COLUMN = 28;

  std::vector<std::vector<sf::CircleShape>> neurons;
  int currentLayerNodeNoStartsAt = 0;

  for (size_t i = 0; i < nn.layers.size(); ++i) {
    std::vector<sf::CircleShape> layer_neurons;
    for (int j = 0; j < nn.layers[i]; ++j) {
      sf::CircleShape neuron(NEURON_RADIUS);
      sf::Color neuronColor = NEURON_COLOR;
      neuronColor.a = static_cast<sf::Uint8>(
          nn.neuralValues[currentLayerNodeNoStartsAt + j] * 255);
      neuron.setFillColor(neuronColor);

      float x = (i + 1) * HORIZONTAL_SPACING + offsetX - NEURON_RADIUS +
                (NEURON_RADIUS * 3) * (int)(j / MAX_COUNT_PER_COLUMN);
      int nodesPerRow =
          ((j / MAX_COUNT_PER_COLUMN + 1) * MAX_COUNT_PER_COLUMN) > nn.layers[i]
              ? nn.layers[i] % MAX_COUNT_PER_COLUMN
              : MAX_COUNT_PER_COLUMN;
      float layer_height = WINDOW_HEIGHT / (nodesPerRow + 2);
      float y = ((j % MAX_COUNT_PER_COLUMN) + 1) * layer_height -
                NEURON_RADIUS + offsetY;
      neuron.setPosition(x, y);

      layer_neurons.push_back(neuron);
    }
    currentLayerNodeNoStartsAt += nn.layers[i];
    neurons.push_back(layer_neurons);
  }

  return neurons;
}

static void
drawConnections(sf::RenderWindow &window, const NeuralNetwork &nn,
                const std::vector<std::vector<sf::CircleShape>> &neurons) {
  const float NEURON_RADIUS = 3.f;
  const float HORIZONTAL_SPACING = WINDOW_WIDTH / (nn.layers.size() + 2);
  const sf::Color NEURON_COLOR = sf::Color::White;
  const int MAX_COUNT_PER_COLUMN = 28;

  const sf::Color INACTIVE_COLOR = sf::Color(255, 255, 255, 50);

  for (size_t i = 0; i < neurons.size() - 1; ++i) {
    for (size_t j = 0; j < neurons[i].size(); ++j) {
      for (size_t k = 0; k < neurons[i + 1].size(); ++k) {
        float weight = nn.weights[i][j * neurons[i + 1].size() + k];
        sf::Color lineColor;
        if (weight >= 0) {
          lineColor = sf::Color(255, 0, 0, static_cast<sf::Uint8>(weight * 64));
        } else {
          lineColor =
              sf::Color(0, 255, 0, static_cast<sf::Uint8>(-weight * 64));
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
}

static void
drawNeurons(sf::RenderWindow &window,
            const std::vector<std::vector<sf::CircleShape>> &neurons) {
  for (const auto &layer : neurons) {
    for (const auto &neuron : layer) {
      window.draw(neuron);
    }
  }
}

void drawNeuralNetwork(sf::RenderWindow &window, const NeuralNetwork &nn,
                       float offsetX, float offsetY) {
  auto neurons = createNeuronsForRendering(nn, offsetX, offsetY);
  drawConnections(window, nn, neurons);
  drawNeurons(window, neurons);
}
