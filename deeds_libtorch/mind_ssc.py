def mind_ssc(image, quantisation_step):
    return image

import torch
import torch.nn as nn
import torch.nn.functional as F

def pdist(x, p=2):
    if p==1:
        dist = torch.abs(x.unsqueeze(2) - x.unsqueeze(1)).sum(dim=3)
    elif p==2:
        xx = (x**2).sum(dim=2).unsqueeze(2)
        yy = xx.permute(0, 2, 1)
        dist = xx + yy - 2.0 * torch.bmm(x, x.permute(0, 2, 1))
        dist[:, torch.arange(dist.shape[1]), torch.arange(dist.shape[2])] = 0
    return dist

def smooth(img, sigma):
    device = img.device

    sigma = torch.tensor([sigma]).to(device)
    N = torch.ceil(sigma * 3.0 / 2.0).long().item() * 2 + 1

    weight = torch.exp(-torch.pow(torch.linspace(-(N // 2), N // 2, N).to(device), 2) / (2 * torch.pow(sigma, 2) + 1e-6))
    weight /= weight.sum()

    img = filter1D(img, weight, 0)
    img = filter1D(img, weight, 1)
    img = filter1D(img, weight, 2)

    return img

def filter1D(img, weight, dim, padding_mode='replicate'):
    B, C, D, H, W = img.shape
    N = weight.shape[0]

    padding = torch.zeros(6,)
    padding[[4 - 2 * dim, 5 - 2 * dim]] = N//2
    padding = padding.long().tolist()

    view = torch.ones(5,)
    view[dim + 2] = -1
    view = view.long().tolist()

    return F.conv3d(F.pad(img.view(B*C, 1, D, H, W), padding, mode=padding_mode), weight.view(view)).view(B, C, D, H, W)

def mind_ssc_deeds(img, delta=1, sigma=0.8):
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [0, 1, 0],
                                      [delta, 0, 1],
                                      [delta, 1, 2],
                                      [delta, 1, 1],
                                      [delta, 2, 1]])
def mind_ssc(img, delta=1, sigma=0.8):
    # see http://mpheinrich.de/pub/miccai2013_943_mheinrich.pdf for details on the MIND-SSC descriptor
    img = img.unsqueeze(0).unsqueeze(0)
    device = img.device

    # define start and end locations for self-similarity pattern
    six_neighbourhood = torch.tensor([[0, 1, 1],
                                      [1, 1, 0],
                                      [1, 0, 1],
                                      [1, 1, 2],
                                      [2, 1, 1],
                                      [1, 2, 1]], dtype=torch.float, device=device)

    # squared distances
    dist = pdist(six_neighbourhood.unsqueeze(0)).squeeze(0)

    # define comparison mask
    x, y = torch.meshgrid(torch.arange(6, device=device), torch.arange(6, device=device))
    mask = ((x > y).view(-1) & (dist == 2).view(-1)) # twelve neighbourhood (connection matrix of 6 nodes)

    # build kernel
    idx_shift1 = six_neighbourhood.unsqueeze(1).repeat(1,6,1).view(-1,3)[mask, :].long()
    idx_shift2 = six_neighbourhood.unsqueeze(0).repeat(6,1,1).view(-1,3)[mask, :].long()
    kernel_mind_patch_a_select = torch.zeros((12, 1, 3, 3, 3), device=device)
    kernel_mind_patch_a_select.view(-1)[torch.arange(12, device=device) * 27 + idx_shift1[:,0] * 9 + idx_shift1[:, 1] * 3 + idx_shift1[:, 2]] = 1
    # kernel_mind_patch_a_select/2 each contain 12 patch coordinates (for the difference 24 patches are needed I(12) - I(12) resulting in 12 edge differences)
    # (kernel_mind_patch_b_select+kernel_mind_patch_a_select).permute(1,2,3,4,0) will yield ssc features: per column 2 patches are invoked to calculate a feature
    kernel_mind_patch_b_select = torch.zeros((12, 1, 3, 3, 3), device=device)
    kernel_mind_patch_b_select.view(-1)[torch.arange(12, device=device) * 27 + idx_shift2[:,0] * 9 + idx_shift2[:, 1] * 3 + idx_shift2[:, 2]] = 1
    rpad = nn.ReplicationPad3d(delta)

    # compute patch-ssd
    mind_selected_patches_a = F.conv3d(rpad(img), kernel_mind_patch_a_select, dilation=delta)
    # convolute kernel with padded version of image (3)^n = 3^3 = 27 steps (for every voxel calculate patch distance from this voxels patch to adjacent patches)
    mind_selected_patches_b = F.conv3d(rpad(img), kernel_mind_patch_b_select, dilation=delta) #
    unsmoothed_mind = ((mind_selected_patches_a - mind_selected_patches_b) ** 2) # same as w1 in deeds code but for 12-neighbourhood
    # ssd = smooth(unsmoothed_mind, sigma)
    ssd = unsmoothed_mind
    # MIND equation
    mind = ssd - torch.min(ssd, 1, keepdim=True)[0]
    # mind_var = torch.max(torch.mean(mind, 1, keepdim=True), 1e-6*torch.ones_like(mind))
    # mind_var = torch.clamp(mind_var, mind_var.mean() * 0.001, mind_var.mean() * 1000)
    # mind /= mind_var
    # mind = torch.exp(-mind)

    #permute to have same ordering as C++ code
    mind = mind[:, torch.tensor([6, 8, 1, 11, 2, 10, 0, 7, 9, 4, 5, 3], dtype=torch.long), :, :, :]

    return mind