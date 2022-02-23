import os
import unittest
import torch

import time
import timeit
import numpy as np
import nibabel as nib

from deeds_libtorch.mind_ssc import mind_ssc, filter1D


from __init__ import SRC_DIR, BUILD_DIR, BUILD_JIT_DIR, CPP_APPLY_BCV_MODULE, CPP_DEEDS_MODULE, test_equal_tensors, log_wrapper

def extract_features(binaries):
    D,H,W,DIGITS = binaries.shape
    sums = torch.empty((D,H,W,12))
    for feature_idx, d_start_idx in enumerate(range(0,DIGITS,5)):
        sums[...,feature_idx] = binaries[..., d_start_idx:d_start_idx+5].sum(dim=-1)
    return sums

class TestMindSsc(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def test_mind_ssc(self):

        #########################################################
        # Prepare inputs
        QUANTISATION_STEP = 1
        D, H, W =  3, 3, 3

        ## Generate some artificial displacements for x,y,z
        image = torch.zeros((D,H,W))
        image[1,1,1] = 5
        #########################################################
        # Get cpp output
        cpp_mind_ssc_long_long, cpp_mind_ssc_twelve = CPP_DEEDS_MODULE.mind_ssc_descriptor(
            image,
            torch.tensor([QUANTISATION_STEP], dtype=torch.int)
        )
        cpp_mind_ssc = cpp_mind_ssc_twelve.permute(-1,0,1,2)

        #########################################################
        # Get torch output
        torch_mind_ssc = mind_ssc(image, QUANTISATION_STEP, sigma=0.0)

        torch_mind_ssc = torch_mind_ssc.reshape(12,D,H,W)
        # def binary(x, bits):
        #     mask = 2**torch.arange(bits).to(x.device, x.dtype)
        #     return x.unsqueeze(-1).bitwise_and(mask).ne(0).byte()
        # BITS_PER_FEATURE = 5
        # NUM_FEAURES = 12
        # binaries = binary(cpp_mind_ssc, 64)[...,:BITS_PER_FEATURE*NUM_FEAURES]
        # features = extract_features(binaries)
        #########################################################
        # Assert difference
        assert test_equal_tensors(cpp_mind_ssc, torch_mind_ssc)

if __name__ == '__main__':
    # unittest.main()
    tests = TestMindSsc()
    tests.test_mind_ssc()
