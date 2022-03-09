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
    patch_length, hw, hw_dilation, alpha, inner_patch_dilation=None):
    assert feature_volume_a.shape == feature_volume_b.shape
    # hw == search radius

    if not inner_patch_dilation is None:
        raise NotImplementedError("Voxel-wise hw_dilation is not implemented yet.")
    else:
        vox_hw_dilation = torch.tensor([1,1,1])

    C_FEAT = feature_volume_a.shape[0]
    SPATIAL = torch.tensor(feature_volume_a.shape[-3:])

    patch_length = torch.tensor([patch_length, patch_length, patch_length])
    PD, PH, PW = (SPATIAL/patch_length).floor().int()
    # patch_length  = (SPATIAL/patch_length).ceil().int()
    D_CONV_PAD, H_CONV_PAD, W_CONV_PAD = (SPATIAL/patch_length).ceil()*patch_length-SPATIAL
    # Add batch dimension
    feature_volume_a = feature_volume_a.unsqueeze(0)
    feature_volume_b = feature_volume_b.unsqueeze(0)

    # conv_a_pad_lens = [
    #     (D_CONV_PAD/2).floor().int().item(), (D_CONV_PAD/2).ceil().int().item(),
    #     (H_CONV_PAD/2).floor().int().item(), (H_CONV_PAD/2).ceil().int().item(),
    #     (W_CONV_PAD/2).floor().int().item(), (W_CONV_PAD/2).ceil().int().item()
    # ]

    # conv_b_pad_lens = [
    #     (D_CONV_PAD/2).floor().int().item(), (D_CONV_PAD/2).ceil().int().item(),
    #     (H_CONV_PAD/2).floor().int().item(), (H_CONV_PAD/2).ceil().int().item(),
    #     (W_CONV_PAD/2).floor().int().item(), (W_CONV_PAD/2).ceil().int().item()
    # ]

    # pad_conv_a = torch.nn.ReplicationPad3d(conv_a_pad_lens)
    # pad_conv_b = torch.nn.ReplicationPad3d(conv_b_pad_lens)
    # feature_volume_a = pad_conv_a(feature_volume_a)
    # feature_volume_b = pad_conv_b(feature_volume_b)

    search_width = 2*hw+1
    search_labelcount = search_width**3
    dilated_search_width = 2*hw*hw_dilation+1
    dilated_expanded_labelcount = dilated_search_width**3

    pad_b = torch.nn.ReplicationPad3d((hw*hw_dilation,)*6)
    feature_volume_b = pad_b(feature_volume_b)

    # This kernel reduces the first feature image to patch dimensions
    reducing_kernel = torch.ones([1, C_FEAT] + patch_length.tolist()) # [C_OUT, C_IN, KD, KH, KW]

    # This kernel builds the search positions e.g.
    # [[0,0,0],
    #  [0,0,1],
    #  [0,0,0]] if right neighbour shall be substracted (vox shift==1, 2D example)

    offset_kernel =  (
        torch.eye(search_labelcount)
        .view([search_labelcount, search_width, search_width, search_width])
    )

    # Dilated kernel looks like
    # With hw_dilation==1 the kernel will be e.g.
    # [[0,0,0,0,0],
    #  [0,0,0,0,0],
    #  [0,0,0,0,1],
    #  [0,0,0,0,0],
    #  [0,0,0,0,0]] if right neighbour shall be substracted (vox shift==1, hw_dilation==1, 2D example)

    dilated_kernel = torch.zeros([search_labelcount, dilated_search_width, dilated_search_width, dilated_search_width])
    dilated_kernel[:,::hw_dilation,::hw_dilation,::hw_dilation] = offset_kernel

    offset_k_len = patch_length*dilated_search_width

    # Now expand the kernel to full voxel-patch dimensions to capture every voxel in the patches
    vox_expanded_kernel = (
        dilated_kernel
        .repeat_interleave(patch_length[-3], dim=-3)
        .repeat_interleave(patch_length[-2], dim=-2)
        .repeat_interleave(patch_length[-1], dim=-1)
        .unsqueeze(1)
        .expand([search_labelcount, C_FEAT] + offset_k_len.tolist())
    ) # [C_OUT, C_IN, KD, KH, KW]

    if vox_expanded_kernel.shape[-1]%2 == 0:
        even_kernel_pad = torch.stack(
            [(patch_length-1)%2*patch_length,
            torch.zeros(3)], dim=-1
        ).int().view(-1).tolist()
        pad_conv_b = torch.nn.ReplicationPad3d(even_kernel_pad)
        feature_volume_b = pad_conv_b(feature_volume_b)
    # TODO: Implement vox_hw_dilation_x, vox_hw_dilation_y, vox_hw_dilation_z of CPP code controlled by RAND_SAMPLES variable
    # This adds hw_dilation to the expanded voxel grid (not implemented since voxel hw_dilation is always 1 in current deeds code)

    accumulated_p_a = torch.nn.functional.conv3d(feature_volume_a, reducing_kernel, stride=patch_length.tolist())
    accumulated_p_b = torch.nn.functional.conv3d(feature_volume_b, vox_expanded_kernel, stride=patch_length.tolist())

    datacost = (accumulated_p_a - accumulated_p_b).abs()

    alphai = 1/alpha
    patch_norm = patch_length[0]/hw_dilation
    vox_norm = (patch_length/vox_hw_dilation).ceil().prod()

    alpha_unary=0.5*alphai*patch_norm/vox_norm

    return alpha_unary*datacost.view(search_width, search_width, search_width, PD, PH, PW)

def warpImageCL():
    raise NotImplementedError()