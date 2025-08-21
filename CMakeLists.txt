cmake_minimum_required(VERSION 3.10)
project(main_project)



# Boost
# set(Boost_INCLUDEDIRS "specify if needed")
# set(Boost_ROOT "specify if needed")
# set(Boost_DIR "specify if needed")
find_package(Boost COMPONENTS REQUIRED program_options filesystem)
message(STATUS "Boost_INCLUDE_DIRS: ${Boost_INCLUDE_DIRS}")
if (Boost_FOUND)
    include_directories(${Boost_INCLUDE_DIR})
else()
    message(FATAL_ERROR "no Boost supprot")
endif()


# OpenMP
find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
else()
    message(FATAL_ERROR "no OpenMP supprot")
endif()

add_subdirectory(flat) # builds ivf_flat library
add_subdirectory(pq) # builds ivf_flat library
add_subdirectory(testing) #builds testing 