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
    mov_img=read_Nifti(args.moving)#img_tensor of size(D,H,W)
    D,H,W=mov_img.shape

    #reading affine matrix
    if(args.affine_mat):
        print("----Reading affine matrix---")
        X_np=read_File(args.affine_mat)#1d list-reshape to(3,4)
        X=torch.from_numpy(X_np)#converted in to tensor
        X.reshape(3,4)
    else:
        print("---Using identity transform----")
        X=torch.eye(4,3)#matrix for identity transform

    #reading displacement field
    print("---Reading displacement field---")
    disp_field_np=read_File(args.disp_field)#1d list
    disp_field=torch.from_numpy(disp_field_np)
    #disp_field.view(D,H,W)#reshaping in to (D,H,W)

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
    D1=D/grid_step
    H1=H/grid_step
    W1=W/grid_step
    sz=D1*H1*W1
    u1=torch.zeros(sz)
    v1=torch.zeros(sz)
    w1=torch.zeros(sz)
    for i in range(sz):
        u1[i]=disp_field[i]
        v1[i]=disp_field[i+sz]
        w1[i]=disp_field[i+sz*2]
    u1.view(D1,H1,W1)
    v1.view(D1,H1,W1)
    w1.view(D1,H1,W1)

    #doing Upsampling for the field
    u1_,v1_,w1_=upsampleDeformationsCL(ux,vx,wx,u1,v1,w1)

    #warping segmentation
    warped_seg=warpAffineS(mov_img,X,u1_,v1_,w1_)

    #writing the warped niftii file
    warped_np=warped_seg.numpy()
    warped_nii=nib.Nifti2Image(warped_np,affine='None')#to check
    f=open(args.deformed,'w')
    f.write(warped_nii)
    f.close()

if __name__ == '__main__':
    main(sys.argv[1:])
