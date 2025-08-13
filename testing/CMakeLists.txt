cmake_minimum_required(VERSION 3.10)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)


include_directories(../../serf/include)
include_directories(../../serf/include/common)
include_directories(../../serf/src)
include_directories(../../src)

include_directories(../../irangegraph/include)

# Set path to the ANNS static library
set(ANNS_LIB_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../build/src/libANNS.a)

add_executable(test_ivf_flat test_ivf_flat.cpp)
target_link_libraries(test_ivf_flat PRIVATE ivf_flat ${ANNS_LIB_PATH} Boost::program_options)