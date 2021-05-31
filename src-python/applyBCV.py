import sys
import argparse
import torch

def main():
    torch.ops.load_library("build/libapplyBCV.dylib")
    torch.ops.deeds_applyBCV.applyBCV_main(len(sys.argv), sys.argv)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('-M', '--moving', type=str, help="mooving.nii.gz path")
    parser.add_argument('-O', '--out_prefix', type=str, help="Output prefix")
    parser.add_argument('-D', '--deformed', type=str, help="deformed.nii.gz path")
    parser.add_argument('-A', '--affine-mat', type=str, help="affine_matrix.txt path")
    args = parser.parse_args()
    main()
