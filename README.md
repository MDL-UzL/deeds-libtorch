# torch-deedsBCV
Source deedsBCV see https://github.com/mattiaspaul/deedsBCV

# Installation
1. Install ITKsnap: http://www.itksnap.org/pmwiki/pmwiki.php
1.1 Export the application binary path to in your .zshrc or .bashrc to have access to the "c3d" tool (http://www.itksnap.org/pmwiki/pmwiki.php?n=Convert3D.Convert3D) by using `export PATH="/Applications/ITK-SNAP.app/Contents/bin:$PATH"`

1. Install your favorite editor (my recommendation is: VSCode) for x86_64. In case you are using M1 powered MacBook you can optionally install an arm toolchain.
1. Install homebrew package manager for x86_64 and arm (see https://soffes.blog/homebrew-on-apple-silicon)
1. brew install cmake
1. brew install zlib
1. brew install libomp
1. Install miniconda or conda for your python environments
1.1 brew install --cask miniconda

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

# Data
Development test data can be found in
``