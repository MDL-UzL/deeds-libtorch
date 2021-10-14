cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
project(deeds-libtorch)

list(APPEND CMAKE_PREFIX_PATH "./lib/libtorch")
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)

# Define our library target
add_library(libtorch-applyBCV SHARED src/applyBCV.cpp)
# Enable C++14
target_compile_features(libtorch-applyBCV PRIVATE cxx_std_14)
# Link against LibTorch
target_link_libraries(libtorch-applyBCV "${TORCH_LIBRARIES}")
# Link against ZLIB
target_link_libraries(libtorch-applyBCV ZLIB::ZLIB)

# Build original binaries
add_executable(deedsBCV src/deedsBCV0.cpp)
target_compile_features(deedsBCV PRIVATE cxx_std_14)
target_link_libraries(deedsBCV ZLIB::ZLIB)

add_executable(linearBCV src/linearBCV.cpp)
target_compile_features(linearBCV PRIVATE cxx_std_14)
target_link_libraries(linearBCV ZLIB::ZLIB)

add_executable(applyBCV src/applyBCV.cpp)
target_compile_features(applyBCV PRIVATE cxx_std_14)
target_link_libraries(applyBCV ZLIB::ZLIB)
target_link_libraries(applyBCV "${TORCH_LIBRARIES}")

add_executable(applyBCVfloat src/applyBCVfloat.cpp)
target_compile_features(applyBCVfloat PRIVATE cxx_std_14)
target_link_libraries(applyBCVfloat ZLIB::ZLIB)
target_link_libraries(applyBCVfloat "${TORCH_LIBRARIES}")