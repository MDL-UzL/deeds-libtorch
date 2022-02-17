import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def warpAffineS(image_seg, affine_mat, u1, v1, w1):

    D, H, W = image_seg.shape

    affine_mat[2,:], affine_mat[0,:] = affine_mat.clone()[0,:], affine_mat.clone()[2,:]
    affine_mat[:,2], affine_mat[:,0] = affine_mat.clone()[:,0], affine_mat.clone()[:,2]

    # Rescale displacements
    affine_mat[0,-1] = (affine_mat[0,-1]/W)*2.0*affine_mat[0,0]
    affine_mat[1,-1] = (affine_mat[1,-1]/H)*2.0*affine_mat[1,1]
    affine_mat[2,-1] = (affine_mat[2,-1]/D)*2.0*affine_mat[2,2]

    affine_mat[0,-1] = affine_mat[0,:4].sum()-1.0
    affine_mat[1,-1] = affine_mat[1,:4].sum()-1.0
    affine_mat[2,-1] = affine_mat[2,:4].sum()-1.0

    grid_all = F.affine_grid(affine_mat.unsqueeze(0),(1,1,D,H,W),align_corners=True)#affine_matransformation grid
    # grid_all-=1.0
    # grid_all[...,0]-=u_disp
    # +(affine_mat[1,-1]/H)*2.0
# +(affine_mat[2,-1]/D)*2.0
    warped = F.grid_sample(image_seg.unsqueeze(0).unsqueeze(0).float(), grid_all, mode='nearest', align_corners=True, padding_mode='border')

    # u1 = u1.unsqueeze(0).unsqueeze(0)
    # v1 = v1.unsqueeze(0).unsqueeze(0)
    # w1 = w1.unsqueeze(0).unsqueeze(0)

    # #combining displacement field
    # w1 = u1/W/2.
    # v1 = v1/H/2.
    # u1 = w1/D/2.
    # u1, v1, w1 = w1, v1, u1
    # disp_uvw=torch.cat((u1,v1,w1),dim=-1).view(1,D,H,W,3)#combined displacemnt_field
    # warp_grid+=disp_uvw #Warp grid for segmentation
    #reshaping the input flow field to shape(N,C,D,H,W),N=C=1

    return warped.squeeze(0).squeeze(0).short().permute(2,1,0)

def interp3xyz():
    raise NotImplementedError()

def interp3xyzB():
    raise NotImplementedError()

def dataCostCL():
    raise NotImplementedError()

def warpImageCL():
    raise NotImplementedError()

def warpAffine():
    raise NotImplementedError()