cmake_minimum_required(VERSION 3.10)
project(ivf_flat_repository)

find_package(OpenMP REQUIRED)


add_compile_options(-march=native)


# Set the C++ standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

set(CMAKE_BUILD_TYPE Debug)

# Disable GPU and Python bindings
add_definitions(-DFAISS_NO_GPU)
add_definitions(-DFAISS_NO_PYTHON)

# Include directories for FAISS and other project components
include_directories(
    ${CMAKE_SOURCE_DIR}/faiss
    ${CMAKE_SOURCE_DIR}/faiss/build
    ${CMAKE_SOURCE_DIR}/build/faiss
    ${CMAKE_SOURCE_DIR}/../include
)

# Source files
set(SRC_FILES
    flat/ivf_flat.cpp
    ../src/distance.cpp
    ../src/storage.cpp
    flat/testing.cpp
)

find_library(FAISS_LIB faiss PATHS ${CMAKE_SOURCE_DIR}/faiss/build/faiss)

# Create executable
add_executable(benchmark ${SRC_FILES})

target_link_libraries(benchmark PRIVATE ${FAISS_LIB} OpenMP::OpenMP_CXX -lopenblas)

# Link required libraries
find_package(Threads REQUIRED)
target_link_libraries(benchmark PRIVATE Threads::Threads)

# Recommended compile options
target_compile_options(benchmark PRIVATE -O3 -Wall -Wextra)
