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

def calc_datacost(feature_volume_a, feature_volume_b,
    grid_divisor, hw, stride, alpha):
    assert feature_volume_a.shape == feature_volume_b.shape

    C_FEAT = feature_volume_a.shape[0]
    SPATIAL = torch.tensor(feature_volume_a.shape[-3:])

    patch_length = (SPATIAL/grid_divisor).floor().int()
    PD, PH, PW = (SPATIAL/patch_length).ceil().int()
    D_CONV_PAD, H_CONV_PAD, W_CONV_PAD = (SPATIAL/patch_length).ceil()*patch_length-SPATIAL
    # Add batch dimension
    feature_volume_a = feature_volume_a.unsqueeze(0)
    feature_volume_b = feature_volume_b.unsqueeze(0)

    conv_pad_lens = [
        (D_CONV_PAD//2).int().item(), (D_CONV_PAD/2).ceil().int().item(),
        (H_CONV_PAD//2).int().item(), (H_CONV_PAD/2).ceil().int().item(),
        (W_CONV_PAD//2).int().item(), (W_CONV_PAD/2).ceil().int().item()
    ]
    pad_conv = torch.nn.ReplicationPad3d(conv_pad_lens)
    feature_volume_a = pad_conv(feature_volume_a)
    feature_volume_b = pad_conv(feature_volume_b)

    search_width = 2*hw+1
    search_labelcount = search_width**3

    pad_b = torch.nn.ReplicationPad3d((hw*stride,)*6)
    feature_volume_b = pad_b(feature_volume_b)

    # Conv3d(12, out_channels, kernel_size)
    # NON offset image
    # IN 12,1,D,H,W
    # OUT 12,S_POS,PD,PH,PW
    # KERNEL 1,S_POS,PATCH_LEN,PATCH_LEN,PATCH_LEN
    reducing_kernel = torch.ones([1, C_FEAT] + patch_length.tolist()) # [C_OUT, C_IN, KD, KH, KW]

    # offset image
    # IN 12,1,D+2*PAD,H+2*PAD,W+2*PAD
    # OUT 12,S_POS,PD,PH,PW
    # KERNEL 1,S_POS,PATCH_LEN*2,PATCH_LEN*2,PATCH_LEN*2 [0,1,
    #                                                     0,0] expanded + dilated variants. [0,0,1, 0,0,0] -> 2*HW+1 variants.
    offset_k_len = patch_length*search_width
    offset_kernel = (
        torch.eye(search_labelcount)
        .view([search_labelcount, search_width, search_width, search_width])
        .repeat_interleave(patch_length[-3], dim=-3)
        .repeat_interleave(patch_length[-2], dim=-2)
        .repeat_interleave(patch_length[-1], dim=-1)
        .unsqueeze(1)
        .expand([search_labelcount, C_FEAT] + offset_k_len.tolist())
    ) # [C_OUT, C_IN, KD, KH, KW]
    # TODO: Build kernel with zeros in between in z,y,x direction according to skipz,skipy,skipx

    accumulated_p_a = torch.nn.functional.conv3d(feature_volume_a, reducing_kernel, stride=patch_length.tolist())
    accumulated_p_b = torch.nn.functional.conv3d(feature_volume_b, offset_kernel, stride=patch_length.tolist())

    datacost = (accumulated_p_a - accumulated_p_b).abs()
    return datacost.view(search_width, search_width, search_width, PD, PH, PW)

def warpImageCL():
    raise NotImplementedError()