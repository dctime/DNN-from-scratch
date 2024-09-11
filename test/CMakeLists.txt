cmake_minimum_required(VERSION 3.22.1)

set(CMAKE_CXX_STANDARD 20)

project(dnn-from-scratch-test)

add_executable(${PROJECT_NAME} test.cpp) 

target_link_libraries(${PROJECT_NAME} PUBLIC gtest)



