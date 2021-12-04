import sys
import os
import argparse
import torch
from torch.utils.cpp_extension import load

def main(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--moving', type=str, help="moving.nii.gz path")
    parser.add_argument('-O', '--out_prefix', type=str, help="Output prefix")
    parser.add_argument('-D', '--deformed', type=str, help="deformed.nii.gz path")
    parser.add_argument('-A', '--affine-mat', type=str, help="affine_matrix.txt path")
    args = parser.parse_args(argv)

    print(args.moving)
    print(args.out_prefix)
    print(args.deformed)
    print(args.affine_mat)

    # Add implementation

if __name__ == '__main__':
    main(sys.argv)
