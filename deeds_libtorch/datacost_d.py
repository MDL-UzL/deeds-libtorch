import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def warp(moving, affine_mat, w, v, u, mode='nearest'):

    assert mode in ['nearest', 'bilinear']
    if mode == 'bilinear':
        raise NotImplementedError()

    D, H, W = moving.shape
    affine_mat = affine_mat.t()[0:3]
    displacements = torch.stack([w,v,u], dim=-1)

    # Rescale displacements
    displacements = 2.*displacements/(torch.tensor([D-1,H-1,W-1]).view(1,1,1,3))

    # Switch x,z
    affine_mat[2,:], affine_mat[0,:] = affine_mat.clone()[0,:], affine_mat.clone()[2,:]
    affine_mat[:,2], affine_mat[:,0] = affine_mat.clone()[:,0], affine_mat.clone()[:,2]

    # Rescale displacements
    affine_mat[0,-1] = (affine_mat[0,-1]/W)*2.0#*affine_mat[0,0]
    affine_mat[1,-1] = (affine_mat[1,-1]/H)*2.0#*affine_mat[1,1]
    affine_mat[2,-1] = (affine_mat[2,-1]/D)*2.0#*affine_mat[2,2]

    # Compensate sheering and scaling
    affine_mat[2,-1] = affine_mat[2,:4].sum()-1.0
    affine_mat[1,-1] = affine_mat[1,:4].sum()-1.0
    affine_mat[0,-1] = affine_mat[0,:4].sum()-1.0

    # Switch w,v,u -> u,w,v
    displacements[...,2], displacements[...,1], displacements[...,0] = displacements.clone()[...,1], displacements.clone()[...,0], displacements.clone()[...,2]

    grid_all = F.affine_grid(affine_mat.unsqueeze(0),(1,1,D,H,W),align_corners=True)
    grid_all += displacements
    warped = F.grid_sample(moving.unsqueeze(0).unsqueeze(0).float(), grid_all, mode=mode, align_corners=True, padding_mode='border')

    warped = warped.squeeze(0).squeeze(0)
    if mode == 'nearest':
        return warped.short()
    elif mode == 'bilinear':
        return warped
    else:
        raise ValueError()

def warpAffineS(label, affine_mat, w, v, u):
    return warp(label, affine_mat, w, v, u, mode='nearest')

def warpAffine(image, affine_mat, w, v, u):
    # return warp(image, affine_mat, w, v, u, mode='bilinear')
    D, H, W = image.shape
    affine_mat = affine_mat.t()[0:3]
    displacements = torch.stack([w,v,u], dim=-1)

    # Rescale displacements
    displacements = 2.*displacements/(torch.tensor([D-1,H-1,W-1]).view(1,1,1,3))

    # # Switch x,z
    # affine_mat[2,:], affine_mat[0,:] = affine_mat.clone()[0,:], affine_mat.clone()[2,:]
    # affine_mat[:,2], affine_mat[:,0] = affine_mat.clone()[:,0], affine_mat.clone()[:,2]

    # Rescale displacements
    affine_mat[0,-1] = (affine_mat[0,-1]/W)*2.0#*affine_mat[0,0]
    affine_mat[1,-1] = (affine_mat[1,-1]/H)*2.0#*affine_mat[1,1]
    affine_mat[2,-1] = (affine_mat[2,-1]/D)*2.0#*affine_mat[2,2]

    # Compensate sheering and scaling
    affine_mat[2,-1] = affine_mat[2,:4].sum()-1.0
    affine_mat[1,-1] = affine_mat[1,:4].sum()-1.0
    affine_mat[0,-1] = affine_mat[0,:4].sum()-1.0

    # Switch
    displacements[...,0], displacements[...,1], displacements[...,2] = displacements.clone()[...,1], displacements.clone()[...,0], displacements.clone()[...,2] #1,2,0

    grid_all = F.affine_grid(affine_mat.unsqueeze(0),(1,1,D,H,W),align_corners=True)
    grid_all += displacements
    warped = F.grid_sample(image.unsqueeze(0).unsqueeze(0).float(), grid_all, mode='bilinear', align_corners=True, padding_mode='border')

    warped = warped.squeeze(0).squeeze(0)

    return warped


def interp3xyz():
    raise NotImplementedError()

def interp3xyzB():
    raise NotImplementedError()

def dataCostCL():
    raise NotImplementedError()

def warpImageCL():
    raise NotImplementedError()