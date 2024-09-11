#include <SDL2/SDL_events.h>
#include <SDL2/SDL_video.h>
#include <iostream>
#include <glad/glad.h>
#include <SDL2/SDL.h>

int main() {
  std::cout << "Hello World" << std::endl;

  if (SDL_Init(SDL_INIT_VIDEO) < 0) {
    std::cout << "Failed to open a window." << std::endl;
    return -1;
  }

  SDL_Window* window = SDL_CreateWindow("Simple DNN", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1080, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);

  bool windowRunning = true;
  while (windowRunning) {
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        windowRunning = false;
      }
    }
    
  }

  SDL_DestroyWindow(window);
  SDL_Quit();
}
