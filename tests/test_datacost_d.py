import os
import unittest
from pathlib import Path
import torch
import timeit
import numpy as np
from bitarray import bitarray
import struct

from __init__ import CPP_APPLY_BCV_MODULE, CPP_DEEDS_MODULE, test_equal_tensors, log_wrapper, mind_compress_features, mind_extract_features, mind_twelve_to_long, unpackbits, packbits
from deeds_libtorch.datacost_d import warpAffineS, warpAffine, calc_datacost

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
        ALPHA = torch.tensor(2.0)

        DILATION = torch.tensor(1).int()
        HW = torch.tensor(1).int()
        PATCH_LENGTH = torch.tensor(1).int()
        D,H,W = 2,2,2

        BIT_VALS = True
        if BIT_VALS:
            mind_image_a = 0*torch.ones(D*H*W).reshape(D,H,W).float()
            mind_image_b = 1*torch.ones(D*H*W).reshape(D,H,W).float()
            fill_val_a = 0b0000_10000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000
            fill_val_b = 0b0
            packed_long_a = mind_image_a.long().fill_(fill_val_a)
            packed_long_b = mind_image_b.long().fill_(fill_val_b)
            packed_long_b[0,0,0] = 0b0000_10000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000_00000
            mind_image_a = mind_extract_features(unpackbits(packed_long_a, 64)).transpose(0,-1)
            mind_image_b = mind_extract_features(unpackbits(packed_long_b, 64)).transpose(0,-1)
            # repacked_a = packbits(binaries_a, torch.long)
        else:
            mind_image_a = 0*torch.ones(D*H*W*12).reshape(12,D,H,W).float()
            mind_image_b = 1*torch.ones(D*H*W*12).reshape(12,D,H,W).float()
            packed_long_a = mind_twelve_to_long(mind_image_a.transpose(0,-1))
            packed_long_b = mind_twelve_to_long(mind_image_b.transpose(0,-1))

        #########################################################
        # Get cpp output
        cpp_costs = log_wrapper(CPP_DEEDS_MODULE.datacost_d_datacost_cl, packed_long_a, packed_long_b, PATCH_LENGTH, HW, DILATION, ALPHA)
        cpp_costs = cpp_costs.permute(3,0,1,2)

        #########################################################
        # Get torch output
        torch_costs = log_wrapper(calc_datacost, mind_image_a, mind_image_b, PATCH_LENGTH.item(), HW, DILATION.item(), ALPHA)

        *_, PD, PH, PW = torch_costs.shape
        torch_costs = (torch_costs.view(-1,PD,PH,PW))
        # Permute y,x in torch tensor and move search features to the front.
        assert test_equal_tensors(cpp_costs, torch_costs), "Tensors do not match"

    def test_warpImageCL(self):
        assert False



if __name__ == '__main__':
    # unittest.main()
    tests = TestDatacostD()
    # tests.test_warpAffineS()
    # tests.test_warpAffine()
    tests.test_dataCostCL()
