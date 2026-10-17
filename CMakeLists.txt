cmake_minimum_required(VERSION 3.18)
project(DemandForecastLSTM CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

file(GLOB SOURCES "${CMAKE_SOURCE_DIR}/src/*.cpp")

add_executable(forecast ${SOURCES})
target_include_directories(forecast PRIVATE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(forecast "${TORCH_LIBRARIES}")
