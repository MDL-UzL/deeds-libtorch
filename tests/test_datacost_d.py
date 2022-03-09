import os
import unittest
from pathlib import Path
import torch
import timeit
import numpy as np
from bitarray import bitarray
import struct

from __init__ import CPP_APPLY_BCV_MODULE, CPP_DEEDS_MODULE, test_equal_tensors, log_wrapper
from deeds_libtorch.datacost_d import warpAffineS, warpAffine, calc_datacost

def extract_features(binaries):
    BITS_PER_FEATURE = 5
    NUM_FEAURES = 12

    binaries = binaries[...,:BITS_PER_FEATURE*NUM_FEAURES]
    D,H,W,DIGITS = binaries.shape

    sums = torch.empty((D,H,W,12))
    for feature_idx, d_start_idx in enumerate(range(0,DIGITS,5)):
        sums[...,feature_idx] = binaries[..., d_start_idx:d_start_idx+5].sum(dim=-1)
    return sums

def compress_features(feat_tensor):
    bit_str = ""
    for feature_idx, feat_val in enumerate(feat_tensor):
        set_bits = ('1' * feat_val.item()).ljust(5, "0")
        bit_str += set_bits
    bit_str = bit_str.rjust(64, "0")
    return [int(b) for b in bit_str]

def unpackbits(x, bits):
    # https://stackoverflow.com/questions/55918468/convert-integer-to-pytorch-tensor-of-binary-bits
    x = x.long()
    mask = 2**torch.arange(bits).to(x.device, x.dtype)
    return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()

def packbits(data, dtype):
    if dtype == torch.long:
        stype = "L"
    data = data.clone()
    BIT_DIM = data.shape[-1]
    SPATIAL_SHAPE = data.shape[0:-1]
    out = torch.empty(SPATIAL_SHAPE).view(-1).to(dtype)

    for t_idx, bit_entry in enumerate(data.view(-1, BIT_DIM)):
        bita = bitarray("".join([str(b) for b in bit_entry.tolist()]), endian='little')
        bit_entry = struct.unpack(stype, bita)[0]
        out[t_idx] = bit_entry
    return out.reshape(SPATIAL_SHAPE)

def mind_twelve_to_long(mind_tensor):
    MIND_DIM = 12
    assert mind_tensor.shape[-1] == MIND_DIM
    mind_tensor = mind_tensor.clone().long()
    SPATIAL_SHAPE = mind_tensor.shape[0:-1]
    out = torch.empty(SPATIAL_SHAPE).view(-1).long()

    for t_idx, mind_twelve in enumerate(mind_tensor.reshape(-1, MIND_DIM)):
        bits = torch.tensor(compress_features(mind_twelve))
        long_value = packbits(bits, torch.long)
        out[t_idx] = long_value
    return out.reshape(SPATIAL_SHAPE)

class TestDatacostD(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def test_warpAffineS(self):
        #########################################################
        # Prepare inputs
        PAD = 1

        CONTENT = 2
        input_img = torch.nn.functional.pad(torch.ones(CONTENT,CONTENT,CONTENT).short(), [PAD]*6)
        D, H, W = input_img.shape

        ## Generate some artificial displacements for x,y,z
        w = 1.2*torch.ones(D,H,W)
        v = 1.5*torch.ones(D,H,W)
        u = torch.zeros(D,H,W)

        T = torch.tensor([
            [2.0, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        )

        #########################################################
        # Get cpp output
        deeds_warped = log_wrapper(CPP_APPLY_BCV_MODULE.datacost_d_warpAffineS, input_img, T, w, v, u)

        #########################################################
        # Get torch output
        torch_warped = log_wrapper(warpAffineS, input_img, T.t(), w, v, u)
        #########################################################
        # Assert difference
        assert test_equal_tensors(deeds_warped, torch_warped.permute(2,1,0)), "Tensors do not match"

    def test_warpAffine(self):
        #########################################################
        # Prepare inputs
        PAD = 1
        CONTENT = 5
        input_img = torch.nn.functional.pad(torch.ones(CONTENT,CONTENT,CONTENT), [PAD]*6)
        # input_img = torch.arange(CONTENT).float()
        # input_img = torch.stack([input_img]*CONTENT, 0)
        # input_img = input_img.view(1,CONTENT,CONTENT)+input_img.view(CONTENT,CONTENT,1)+input_img.view(CONTENT,1,CONTENT)
        D, H, W = input_img.shape

        ## Generate some artificial displacements for x,y,z
        w = torch.ones(D,H,W)
        v = 1.1*torch.ones(D,H,W)
        u = 1.2*torch.ones(D,H,W)

        T = torch.tensor([
            [0.5, 0.0, 0.0, 0.0],
            [0.2, 2.5, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]]
        )

        #########################################################
        # Get cpp output
        cpp_warped = log_wrapper(CPP_APPLY_BCV_MODULE.datacost_d_warpAffine, input_img, T, w, v, u)

        #########################################################
        # Get torch output
        torch_warped = log_wrapper(warpAffine, input_img, T.t(), w, v, u)
        #########################################################
        # Assert difference
        assert test_equal_tensors(cpp_warped, torch_warped), "Tensors do not match"

    def test_dataCostCL(self):
        ALPHA = torch.tensor(1.0)

        DILATION = torch.tensor(1).int()
        HW = torch.tensor(3).int()
        PATCH_LENGTH = torch.tensor(2).int()
        D,H,W = 9,9,9

        BIT_VALS = True
        if BIT_VALS:
            mind_image_a = 0*torch.ones(D*H*W).reshape(D,H,W).float()
            mind_image_b = 1*torch.ones(D*H*W).reshape(D,H,W).float()
            fill_val_a = 0b0000_10000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000
            fill_val_b = 0b0
            packed_long_a = mind_image_a.long().fill_(fill_val_a)
            packed_long_b = mind_image_b.long().fill_(fill_val_b)
            mind_image_a = extract_features(unpackbits(packed_long_a, 64)).transpose(0,-1)
            mind_image_b = extract_features(unpackbits(packed_long_b, 64)).transpose(0,-1)
            # repacked_a = packbits(binaries_a, torch.long)
        else:
            mind_image_a = 0*torch.ones(D*H*W*12).reshape(12,D,H,W).float()
            mind_image_b = 1*torch.ones(D*H*W*12).reshape(12,D,H,W).float()
            packed_long_a = mind_twelve_to_long(mind_image_a.transpose(0,-1))
            packed_long_b = mind_twelve_to_long(mind_image_b.transpose(0,-1))

        #########################################################
        # Get cpp output
        cpp_costs = log_wrapper(CPP_DEEDS_MODULE.datacost_d_datacost_cl, packed_long_a, packed_long_b, PATCH_LENGTH, HW, DILATION, ALPHA)

        #########################################################
        # Get torch output
        torch_costs = log_wrapper(calc_datacost, mind_image_a, mind_image_b, PATCH_LENGTH.item(), HW, DILATION.item(), ALPHA)

        *_, PD, PH, PW = torch_costs.shape
        torch_costs = (torch_costs.view(-1,PD,PH,PW)).permute(1,2,3,0)
        assert test_equal_tensors(cpp_costs, torch_costs), "Tensors do not match"

    def test_warpImageCL(self):
        assert False



if __name__ == '__main__':
    # unittest.main()
    tests = TestDatacostD()
    # tests.test_warpAffineS()
    # tests.test_warpAffine()
    tests.test_dataCostCL()
