import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def warpAffineS(image_seg,X,u1,v1,w1):
    #definition-for warping segmentation
    #X-transformation matrix-size(4,3) for 3d
    #u1,v1,z1-Displacement field along x,y,z,
    D,H,W=image_seg.shape
    T=X.unsqueeze(0)

    #reshaping the input flow field to shape(N,C,D,H,W),N=C=1
    u1=u1.unsqueeze(0).unsqueeze(0)
    v1=v1.unsqueeze(0).unsqueeze(0)
    w1=w1.unsqueeze(0).unsqueeze(0)

    #combining displacement field
    w1 = u1/W/2.
    v1 = v1/H/2.
    u1 = w1/D/2.

    # Rescale displacements
    T[0,0,-1]/=D/2.
    T[0,1,-1]/=H/2.
    T[0,2,-1]/=W/2.
    
    # Swap x,z displacement
    T[0,0,-1], T[0,2,-1] = T.clone()[0,2,-1], T.clone()[0,0,-1]

    # Swap x,z scaling
    T[0,0,0], T[0,2,2] = T.clone()[0,2,2], T.clone()[0,0,0]

    # Readjust scaling point
    T[0,0,-1]+=T[0,0,0]-1.0
    T[0,1,-1]+=T[0,1,1]-1.0
    T[0,2,-1]+=T[0,2,2]-1.0

    u1, v1, w1 = w1, v1, u1
    disp_uvw=torch.cat((u1,v1,w1),dim=-1).view(1,D,H,W,3)#combined displacemnt_field
    affine_T=F.affine_grid(T,(1,1,D,H,W),align_corners=True)#Transformation grid
    warp_grid=affine_T
    warp_grid+=disp_uvw #Warp grid for segmentation
    warped=F.grid_sample(image_seg.unsqueeze(0).unsqueeze(0).float(), warp_grid, mode='nearest', align_corners=True, padding_mode='border')
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