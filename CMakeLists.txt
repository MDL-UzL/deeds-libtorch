set(CMAKE_VERBOSE_MAKEFILE ON)
project(deeds-libtorch)

list(APPEND CMAKE_PREFIX_PATH "./lib/libtorch")
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)

find_package(Torch REQUIRED)
find_package(ZLIB REQUIRED)
# find_package(OpenMP REQUIRED) # OPENMP does currently not work for mac M1

# SET(CMAKE_CXX_FLAGS "-O3 -mavx2 -msse4.2") # one of these create illegal hardware instructions

# Define executable targets
set(executables_list
    linearBCV
    deedsBCV
    applyBCV
    applyBCVfloat
)

set(executables_src_list
    src/linearBCV.cpp
    src/deedsBCV0.cpp
    src/applyBCV.cpp
    src/applyBCVfloat.cpp
)

list(LENGTH executables_list len_executables)
math(EXPR executables_max_idx "${len_executables} - 1")

foreach(idx RANGE ${executables_max_idx})
    list(GET executables_list ${idx} target_name)
    list(GET executables_src_list ${idx} source_file)
    message(STATUS "Adding executable target ${target_name}  ${source_file}")

    add_executable(${target_name} ${source_file})
    target_compile_features(${target_name} PRIVATE cxx_std_14)
    # Link against LibTorch
    target_link_libraries(${target_name} "${TORCH_LIBRARIES}")
    # Link against ZLIB
    target_link_libraries(${target_name} ZLIB::ZLIB)
    # # Link agains libomp
    # target_link_libraries(${target_name} OpenMP::OpenMP_CXX)

endforeach()

# Define library targets
set(libraries_list
    libtorch-applyBCV
)

set(libraries_src_list
    src/applyBCV.cpp
)

list(LENGTH libraries_list len_libs)
math(EXPR libs_max_idx "${len_libs} - 1")

foreach(idx RANGE ${libs_max_idx})
    list(GET libraries_list ${idx} target_name)
    list(GET libraries_src_list ${idx} source_file)
    message(STATUS "Adding library target ${target_name}  ${source_file}")

    add_library(${target_name} SHARED ${source_file})
    target_compile_features(${target_name} PRIVATE cxx_std_14)
    # Link against LibTorch
    target_link_libraries(${target_name} "${TORCH_LIBRARIES}")
    # Link against ZLIB
    target_link_libraries(${target_name} ZLIB::ZLIB)
    # Link agains libomp
    # target_link_libraries(${target_name} OpenMP::OpenMP_CXX)

endforeach()