import sys
import os
import argparse
import torch
from torch.utils.cpp_extension import load

def main(args):
    pass
    # Add function calls to subroutines

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--moving', type=str, help="moving.nii.gz path")
    parser.add_argument('-O', '--out_prefix', type=str, help="Output prefix")
    parser.add_argument('-D', '--deformed', type=str, help="deformed.nii.gz path")
    parser.add_argument('-A', '--affine-mat', type=str, help="affine_matrix.txt path")
    args = parser.parse_args()
    main(args)
