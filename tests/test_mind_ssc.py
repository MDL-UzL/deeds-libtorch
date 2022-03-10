import os
import unittest
import torch

import time
import timeit
import numpy as np
import nibabel as nib

from deeds_libtorch.mind_ssc import mind_ssc, filter1D


from __init__ import SRC_DIR, BUILD_DIR, BUILD_JIT_DIR, CPP_DEEDS_MODULE, test_equal_tensors, log_wrapper, mind_compress_features, mind_extract_features, mind_twelve_to_long, mind_long_to_twelve

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
        D, H, W =  5, 5, 5

        ## Generate some artificial displacements for x,y,z
        image = torch.ones((D,H,W))
        image[1,1,1] = 2
        image[2,2,2] = -1
        #########################################################
        # Get cpp output
        cpp_mind_ssc_long_long, cpp_mind_ssc_twelve, mind_bare = CPP_DEEDS_MODULE.mind_ssc_descriptor(
            image,
            torch.tensor([QUANTISATION_STEP], dtype=torch.int)
        )
        cpp_mind = mind_long_to_twelve(cpp_mind_ssc_long_long)

        #########################################################
        # Get torch output
        torch_mind_ssc = mind_ssc(image, QUANTISATION_STEP, use_smoothing=False, sigma=0.8, compensation_factor=5.)
        torch_mind_ssc = torch_mind_ssc.reshape(12,D,H,W)

        #########################################################
        # Assert difference
        assert test_equal_tensors(cpp_mind_ssc, torch_mind_ssc)

if __name__ == '__main__':
    # unittest.main()
    tests = TestMindSsc()
    tests.test_mind_ssc()
