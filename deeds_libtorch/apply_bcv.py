import sys

from torch._C import device
import os
import argparse
import torch

from deeds_libtorch.file_io import read_nifti, read_affine_file, save_nifti
from deeds_libtorch.transformations import upsampleDeformationsCL
from deeds_libtorch.datacost_d import warpAffineS
import math
import nibabel as nib
import numpy as np
from timeit import default_timer as timer
import time

def main(argv, mod):
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

    #To run in cuda or cpu
    device=torch.device('cpu')

    # reading the nifti image-segmentation?
    print("---Reading moving image---")
    mov_img, header, affine = read_nifti(args.moving)
    D,H,W = mov_img.shape

    #reading affine matrix
    if(args.affine_mat):
        print("----Reading affine matrix---")
        affine_mat = read_affine_file(args.affine_mat)#1d list-reshape to(3,4)
        print(affine_mat)
    else:
        print("---Using identity transform----")
        affine_mat = torch.eye(4,3)#matrix for identity transform

    #reading displacement field
    print("---Reading displacement field---")
    disp_field = np.fromfile(args.out_prefix+"_displacements.dat", np.float32)#1d list
    disp_field = torch.tensor(disp_field).view(3,D//4,H//4,W//4).to(device)

    #creating sample grid
    grid_size=torch.numel(disp_field)
    sample_sz=grid_size/3
    grid_step=round(math.pow(grid_size/sample_sz,0.3333333))
    print("---Grid step---:",grid_step)

    #creating gridded flow field
    u1 = disp_field[0]
    v1 = disp_field[1]
    w1 = disp_field[2]

    #doing Upsampling for the field
    upsampled_u, upsampled_v, upsampled_w = upsampleDeformationsCL(u1,v1,w1, (D,H,W), USE_CONSISTENT_TORCH=True)
    # upsampled_u, upsampled_v, upsampled_w = torch.load("u_out.pth"), torch.load("v_out.pth"), torch.load("w_out.pth")
    #warping segmentation
    warped_seg = warpAffineS(mov_img, affine_mat, upsampled_u, upsampled_v, upsampled_w).to(device)

    #writing the warped niftii file
    save_nifti(warped_seg, args.deformed, header=header)

if __name__ == '__main__':
    main(sys.argv[1:])
