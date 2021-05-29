cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(deeds-cpp)

list(APPEND CMAKE_PREFIX_PATH "./libtorch")
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# Define our library target
add_library(applyBCV SHARED src/applyBCV.cpp)
# Enable C++14
target_compile_features(applyBCV PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(applyBCV "${TORCH_LIBRARIES}")
# Link against ZLIB
target_link_libraries(applyBCV ZLIB::ZLIB)