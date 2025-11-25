cmake_minimum_required(VERSION 3.14)
project(chess-seresnet 
    VERSION 1.0.0
    DESCRIPTION "Chess bot using SE-ResNet neural network evaluation"
    LANGUAGES CXX
)

# Require C++14 standard
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Build type
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# Compiler flags
set(CMAKE_CXX_FLAGS_DEBUG "-g -O0")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

# Find required packages
find_package(Threads REQUIRED)

# Dependencies paths (can be overridden with -D options)
set(FDEEP_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/frugally-deep/include" 
    CACHE PATH "Path to frugally-deep include directory")
set(FUNCTIONALPLUS_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/FunctionalPlus/include" 
    CACHE PATH "Path to FunctionalPlus include directory")
set(EIGEN_INCLUDE_DIR "${CMAKE_SOURCE_DIR}/eigen" 
    CACHE PATH "Path to Eigen include directory")

# Core library sources
set(CORE_SOURCES
    core/types.cpp
    core/tables.cpp
    core/position.cpp
)

set(CORE_HEADERS
    core/types.h
    core/tables.h
    core/position.h
)

# Create core library
add_library(chess_core STATIC ${CORE_SOURCES} ${CORE_HEADERS})
target_include_directories(chess_core PUBLIC ${CMAKE_SOURCE_DIR})

# Main executable
add_executable(chess-seresnet src/main.cpp)
target_link_libraries(chess-seresnet PRIVATE 
    chess_core
    Threads::Threads
)

# Include directories for frugally-deep and its dependencies
target_include_directories(chess-seresnet PRIVATE
    ${FDEEP_INCLUDE_DIR}
    ${FUNCTIONALPLUS_INCLUDE_DIR}
    ${EIGEN_INCLUDE_DIR}
)

# Installation rules
install(TARGETS chess-seresnet
    RUNTIME DESTINATION bin
)

install(FILES model_ser_mid.json
    DESTINATION share/chess-seresnet
)

# Print configuration summary
message(STATUS "")
message(STATUS "chess-seresnet configuration:")
message(STATUS "  Version:        ${PROJECT_VERSION}")
message(STATUS "  Build type:     ${CMAKE_BUILD_TYPE}")
message(STATUS "  C++ Compiler:   ${CMAKE_CXX_COMPILER}")
message(STATUS "  C++ Standard:   ${CMAKE_CXX_STANDARD}")
message(STATUS "")
