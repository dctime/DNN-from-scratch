cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_STANDARD 20)

project(dnn-from-scratch)

# Find the libzip package
find_package(PkgConfig REQUIRED)
pkg_check_modules(LIBZIP REQUIRED libzip)

# Include directories
include_directories(${LIBZIP_INCLUDE_DIRS})

add_executable(${PROJECT_NAME} 
  main.cpp 
  neural_network.cpp
  render.cpp
) 

target_include_directories(${PROJECT_NAME} PUBLIC 
  ${CMAKE_SOURCE_DIR}/eigen-3.4.0
)

target_link_libraries(${PROJECT_NAME} sfml-graphics sfml-window sfml-system ${LIBZIP_LIBRARIES})

