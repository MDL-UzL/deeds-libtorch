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
        sz=16
        D, H, W = sz, sz, sz

        input_img = torch.arange(D*H*W).view(D,H,W).short()

        ## Generate some artificial displacements for x,y,z
        x_disp_field = torch.zeros(D,H,W)
        y_disp_field = torch.zeros(D,H,W)
        z_disp_field = torch.zeros(D,H,W)
        T = torch.eye(3,4)
        T = T+torch.rand_like(T)*.01


        #########################################################
        # Get cpp output
        deeds_warped = log_wrapper(CPP_APPLY_BCV_MODULE.applyBCV_warpAffineS, input_img, T, x_disp_field, y_disp_field, z_disp_field)

        #########################################################
        # Get torch output
        torch_warped = log_wrapper(warpAffineS, input_img, T, x_disp_field ,y_disp_field, z_disp_field)
        #########################################################
        # Assert difference
        assert test_equal_tensors(deeds_warped, torch_warped.short())

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
    unittest.main()
    # tests = TestDatacostD()
    # tests.test_warpAffineS()
