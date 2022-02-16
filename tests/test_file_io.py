import os
import unittest
from pathlib import Path
import torch

from deeds_libtorch.file_io import read_nifti, read_affine_file
from __init__ import SRC_DIR, BUILD_DIR, BUILD_JIT_DIR, APPLY_BCV_MODULE, test_equal_tensors


class TestFileIo(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def test_read_nifti(self):
        cpp_nifti_data = None
        torch_nifti_data = None
        #########################################################
        # Assert difference
        test_equal_tensors(cpp_nifti_data, torch_nifti_data)



    def test_read_affine_file(self):
        cpp_flow_field = None
        torch_flow_field = read_affine_file('/Users/sreejithkk/Internship/deeds-libtorch/','inputflow.dat')

        #########################################################
        # Assert difference
        test_equal_tensors(cpp_flow_field, torch_flow_field)





if __name__ == '__main__':
    # unittest.main()
    tests = TestFileIo()
    tests.test_interp3()
    tests.test_volfilter()
    tests.test_consistentMappingCL()
    tests.test_upsampleDeformationsCL()
