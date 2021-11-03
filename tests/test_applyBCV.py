# This is no unit test but a function to run applyBCV from python

import sys
import os
import argparse
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

os.environ['USE_JIT_COMPILE'] = '1'
THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def main():

    # Load build output
    src_dir = Path(THIS_SCRIPT_DIR, "../src").resolve()
    build_dir = Path(THIS_SCRIPT_DIR, "../build").resolve()
    build_jit_dir = Path(THIS_SCRIPT_DIR, "../build-jit").resolve()

    build_jit_dir.mkdir(exist_ok=True)

    apply_bcv_source = Path.joinpath(src_dir, "applyBCV.cpp").resolve()
    apply_bcv_dylib = Path.joinpath(build_dir, "liblibtorch-applyBCV.dylib").resolve()

    if os.environ.get('USE_JIT_COMPILE', None) == '1':
        # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
        applyBCV_module = load(name="applyBCV_module", sources=[apply_bcv_source], build_directory=build_jit_dir)

    else:
        # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
        torch.ops.load_library(apply_bcv_dylib)
        applyBCV_module = torch.ops.deeds_applyBCV

    applyBCV_module.applyBCV_main(len(sys.argv), sys.argv)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--moving', type=str, help="mooving.nii.gz path")
    parser.add_argument('-O', '--out_prefix', type=str, help="Output prefix")
    parser.add_argument('-D', '--deformed', type=str, help="deformed.nii.gz path")
    parser.add_argument('-A', '--affine-mat', type=str, help="affine_matrix.txt path")
    args = parser.parse_args()
    main()