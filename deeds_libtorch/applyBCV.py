import sys
sys.path.append('./deeds_libtorch')
import os
import argparse
import torch
from torch.utils.cpp_extension import load
from imageIOgzType import read_Nifti,read_File
from transformations import upsampleDeformationsCL
from datacostD import warpAffineS
import math
import nibabel as nib
import numpy as np

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

    # reading the nifti image-segmentation?
    print("---Reading moving image---")
    mov_img = read_Nifti(args.moving)
    D,H,W = mov_img.shape

    #reading affine matrix
    if(args.affine_mat):
        print("----Reading affine matrix---")
        X_np=read_File(args.affine_mat)#1d list-reshape to(3,4)
        X=torch.from_numpy(X_np)#converted in to tensor
        X=X.reshape(4,4)[:3,:]
    else:
        print("---Using identity transform----")
        X=torch.eye(4,3)#matrix for identity transform

    #reading displacement field
    print("---Reading displacement field---")
    disp_field = np.fromfile(args.out_prefix+"_displacements.dat", np.float32)#1d list
    disp_field = torch.tensor(disp_field).view(3,D//4,H//4,W//4)

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
    warped_seg=warpAffineS(mov_img,X,u1_,v1_,w1_)

    #writing the warped niftii file
    warped_np=warped_seg.numpy()
    warped_nii = nib.Nifti2Image(warped_np, affine=np.eye(4))
    nib.save(warped_nii, args.deformed)

if __name__ == '__main__':
    main(sys.argv[1:])
