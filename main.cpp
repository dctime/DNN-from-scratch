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

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
  SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 1);

  SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);

  SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
  SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);


  SDL_Window* window = SDL_CreateWindow("Simple DNN", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, 1080, 720, SDL_WINDOW_SHOWN | SDL_WINDOW_OPENGL);
  SDL_GLContext context = SDL_GL_CreateContext(window);

  gladLoadGLLoader(SDL_GL_GetProcAddress);

  float positions[6] = {
    -0.5f, -0.5f,
    0.0f, 0.5f,
    0.5f, -0.5f
  };

  unsigned int buffer;
  glad_glGenBuffers(1, &buffer);
  glad_glBindBuffer(GL_ARRAY_BUFFER, buffer);
  glad_glBufferData(GL_ARRAY_BUFFER, 6 * sizeof(float), positions, GL_STATIC_DRAW);

  bool windowRunning = true;
  while (windowRunning) {
    glViewport(0, 0, 1080, 720);
    SDL_Event event;
    while (SDL_PollEvent(&event)) {
      if (event.type == SDL_QUIT) {
        windowRunning = false;
      }
    }

    glad_glClearColor(0.0f, 0.0f, 0.0f, 1.0f);
    glad_glClear(GL_DEPTH_BUFFER_BIT | GL_COLOR_BUFFER_BIT);

    glDrawArrays(GL_TRIANGLES, 0, 3);

    SDL_GL_SwapWindow(window);
    
  }

  SDL_DestroyWindow(window);
  SDL_Quit();
}
