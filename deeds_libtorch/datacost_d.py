import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def warpAffineS(image_seg, affine_mat, u1, v1, w1):

    affine_mat = affine_mat.t()[0:3]
    D, H, W = image_seg.shape

    affine_mat[2,:], affine_mat[0,:] = affine_mat.clone()[0,:], affine_mat.clone()[2,:]
    affine_mat[:,2], affine_mat[:,0] = affine_mat.clone()[:,0], affine_mat.clone()[:,2]

    # Rescale displacements
    affine_mat[0,-1] = (affine_mat[0,-1]/W)*2.0#*affine_mat[0,0]
    affine_mat[1,-1] = (affine_mat[1,-1]/H)*2.0#*affine_mat[1,1]
    affine_mat[2,-1] = (affine_mat[2,-1]/D)*2.0#*affine_mat[2,2]

    # Compensate sheering and scaling
    affine_mat[0,-1] = affine_mat[0,:4].sum()-1.0
    affine_mat[1,-1] = affine_mat[1,:4].sum()-1.0
    affine_mat[2,-1] = affine_mat[2,:4].sum()-1.0

    grid_all = F.affine_grid(affine_mat.unsqueeze(0),(1,1,D,H,W),align_corners=True)
    warped = F.grid_sample(image_seg.unsqueeze(0).unsqueeze(0).float(), grid_all, mode='nearest', align_corners=True, padding_mode='border')

    return warped.squeeze(0).squeeze(0).short()

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