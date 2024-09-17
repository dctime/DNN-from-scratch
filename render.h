#pragma once
#include <SFML/Graphics.hpp>

#include "neural_network.h"

struct ImageRenderer {
  sf::RenderWindow &window;
  sf::Sprite sprite;
  sf::Vector2f topLeftPosition;
  sf::Vector2f size;
  sf::Image image; // Store the image once
  //
  sf::Vector2f getScreenLocation(int x, int y) const;
  float getGrayscaleValue(int x, int y) const;
  void updateImage();
};

void renderImageInWindow(sf::RenderWindow &window, const std::string &imagePath,
                         sf::Vector2f position, sf::Vector2f size,
                         ImageRenderer &renderer);

static std::vector<std::vector<sf::CircleShape>>
createNeuronsForRendering(const NeuralNetwork &nn, float offsetX,
                          float offsetY);

static void
drawConnections(sf::RenderWindow &window, const NeuralNetwork &nn,
                const std::vector<std::vector<sf::CircleShape>> &neurons);

static void
drawNeurons(sf::RenderWindow &window,
            const std::vector<std::vector<sf::CircleShape>> &neurons);

void drawNeuralNetwork(sf::RenderWindow &window, const NeuralNetwork &nn,
                       float offsetX, float offsetY);

