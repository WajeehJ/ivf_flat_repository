cmake_minimum_required(VERSION 3.10)

# This project is just for the ivf_flat library
project(ivf_flat_lib)

find_package(OpenMP REQUIRED)
find_package(Threads REQUIRED)

add_compile_options(-march=native)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_BUILD_TYPE Debug)

add_definitions(-DFAISS_NO_GPU -DFAISS_NO_PYTHON)

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss
    ${CMAKE_CURRENT_SOURCE_DIR}/../faiss/build
    ${CMAKE_CURRENT_SOURCE_DIR}/../build/faiss
    ${CMAKE_CURRENT_SOURCE_DIR}/../../include
)

set(SRC_FILES
    ivf_flat.cpp
    ../../src/distance.cpp
    ../../src/storage.cpp
)

find_library(FAISS_LIB faiss PATHS ${CMAKE_CURRENT_SOURCE_DIR}/../faiss/build/faiss)

# Create a static library
add_library(ivf_flat STATIC ${SRC_FILES})

# Output it to build/ivf_flat/
set_target_properties(ivf_flat PROPERTIES
    ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ivf_flat"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/ivf_flat"
)

# Link dependencies so outside projects only need ivf_flat
target_link_libraries(ivf_flat PRIVATE
    ${FAISS_LIB}
    OpenMP::OpenMP_CXX
    Threads::Threads
    -lopenblas
)

target_compile_options(ivf_flat PRIVATE -O3 -Wall -Wextra)
