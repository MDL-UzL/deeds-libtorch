cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(deeds-cpp)

list(APPEND CMAKE_PREFIX_PATH "/Users/christianweihsbach/code/torch-deedsBCV/libtorch")
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(Torch REQUIRED)

# Define our library target
add_library(d_sigmoid SHARED op.cpp)
# Enable C++14
target_compile_features(d_sigmoid PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(d_sigmoid "${TORCH_LIBRARIES}")
# Link against OpenCV
target_link_libraries(d_sigmoid)