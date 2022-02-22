import os
import unittest
from pathlib import Path
import torch
import timeit

from __init__ import CPP_APPLY_BCV_MODULE, test_equal_tensors, log_wrapper
from deeds_libtorch.datacost_d import warpAffineS



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
        w = torch.ones(D,H,W)
        v = torch.ones(D,H,W)
        u = torch.ones(D,H,W)

        T = torch.tensor([
            [0.5, 0.0, 0.0, 0.0],
            [0.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 2.0, 0.0],
            [2.0, 0.0, 1.0, 1.0]]
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

    def test_interp3xyz(self):
        assert False

    def test_interp3xyzB(self):
        assert False

    def test_dataCostCL(self):
        assert False

    def test_warpImageCL(self):
        assert False

    def test_warpAffine(self):
        assert False



if __name__ == '__main__':
    # unittest.main()
    tests = TestDatacostD()
    tests.test_warpAffineS()
