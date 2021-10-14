# torch-deedsBCV
Source deedsBCV see https://github.com/mattiaspaul/deedsBCV

# Installation
1. Install cmake
1. Install zlib
1. Install miniconda or conda for your python environments

## Install python environment
Create a new conda env and run
`pip install .`

## Download libtorch
Download libtorch for your OS and put the folder next to the README.md file in ./libtorch directory

https://pytorch.org/cppdocs/installing.html

https://pytorch.org/get-started/locally/

! Make sure to use the corresponding libtorch version (e.g. v1.9.1) matching your torch version installed in your conda environment - it will produce a non-descriptive "SegFault" error if versions mismatch.

# Build executables and library on MacOSx
To build the library change into ./build dir and execute cmake:

## Prepare
`mkdir build`
`cd build`

## Configure
`cmake ..`

## Build
`cmake --build .`

If this fails make sure you use Clang x86_64 toolkit (building Mac M1 arm libs is a possible TODO)

The build process will currently output four executables and and .dylib file.
You can execute the following executables in your command line:
./build/deedsBCV
./build/applyBCV
./build/applyBCVfloat
./build/linearBCV

The ./deeds_libtorch folder contains a script called "applyBCV.py" which accesses the .dylib file and lets you run the applyBCV command piping your args through python.

Run by using:
`python ./deeds_libtorch/applyBCV.py`

This is the first entry point for development. You can extend this interface for development or debugging.