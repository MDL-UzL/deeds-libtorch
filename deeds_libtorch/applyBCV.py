import sys
import os
import argparse
import torch
from torch.utils.cpp_extension import load

def main():
    if os.environ.get('USE_JIT_COMPILE', None) == '1':
        # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
        applyBCV_module = load(name="applyBCV_module", sources=["src/applyBCV.cpp"])
        applyBCV_module.applyBCV_main(len(sys.argv), sys.argv)
    else:
        # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
        torch.ops.load_library("build/liblibtorch-applyBCV.dylib")
        torch.ops.deeds_applyBCV.applyBCV_main(len(sys.argv), sys.argv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--moving', type=str, help="mooving.nii.gz path")
    parser.add_argument('-O', '--out_prefix', type=str, help="Output prefix")
    parser.add_argument('-D', '--deformed', type=str, help="deformed.nii.gz path")
    parser.add_argument('-A', '--affine-mat', type=str, help="affine_matrix.txt path")
    args = parser.parse_args()
    main()
