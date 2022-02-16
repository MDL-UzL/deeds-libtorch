import sys

from torch._C import device
import os
import argparse
import torch

from deeds_libtorch.file_io import read_nifti, read_affine_file
from deeds_libtorch.transformations import upsampleDeformationsCL
from deeds_libtorch.datacost_d import warpAffineS
import math
import nibabel as nib
import numpy as np
from timeit import default_timer as timer
import time

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

    #To run in cuda or cpu
    device=torch.device('cpu')

    # reading the nifti image-segmentation?
    print("---Reading moving image---")
    mov_img = read_nifti(args.moving).to(device)
    D,H,W = mov_img.shape

    #reading affine matrix
    if(args.affine_mat):
        print("----Reading affine matrix---")
        X = read_affine_file(args.affine_mat)#1d list-reshape to(3,4)
        print(X)
        X=X[:3,:]
    else:
        print("---Using identity transform----")
        X=torch.eye(4,3)#matrix for identity transform

    #reading displacement field
    print("---Reading displacement field---")
    disp_field = np.fromfile(args.out_prefix+"_displacements.dat", np.float32)#1d list
    disp_field = torch.tensor(disp_field).view(3,D//4,H//4,W//4).to(device)

    #creating sample grid
    grid_size=torch.numel(disp_field)
    sample_sz=grid_size/3
    grid_step=round(math.pow(grid_size/sample_sz,0.3333333))
    print("---Grid step---:",grid_step)

    #initializing flow field-full field
    ux=torch.zeros((D,H,W))
    vx=torch.zeros((D,H,W))
    wx=torch.zeros((D,H,W))

    #creating gridded flow field
    u1 = disp_field[0]
    v1 = disp_field[1]
    w1 = disp_field[2]

    #doing Upsampling for the field
    u1_,v1_,w1_ = upsampleDeformationsCL(ux,vx,wx,u1,v1,w1)

    #warping segmentation
    start_2=time.time()
    warped_seg=warpAffineS(mov_img,X,u1_,v1_,w1_).to(device)
    print('time taken warp_seg: %s sec' %(time.time()-start_2))

    #writing the warped niftii file
    warped_np=warped_seg.numpy()
    warped_nii = nib.Nifti1Image(warped_np, affine=np.eye(4,4))
    nib.save(warped_nii,args.deformed)

if __name__ == '__main__':
    main(sys.argv[1:])
