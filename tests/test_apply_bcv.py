# This is no unit test but a function to run applyBCV from python
import os
import sys
import unittest
from pathlib import Path
import torch
import nibabel as nib
import torch.nn.functional as F

from deeds_libtorch.apply_bcv import main as apply_bcv_main

from __init__ import SRC_DIR, BUILD_DIR, BUILD_JIT_DIR, TEST_DATA_DIR, CPP_APPLY_BCV_MODULE
global u1_,v1_,w1_
class TestApplyBCV(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)



    def test_main_case_two(self):

        TEST_BASE_DIR = Path(TEST_DATA_DIR, "case_2").resolve()
        TEST_OUTPUT_DIR = TEST_BASE_DIR.joinpath("test_output")
        TEST_OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        MOVING_INPUT_FILE = Path(TEST_BASE_DIR, "label_toy_nifti_random_DHW_12x24x36.nii.gz").resolve()
        FLOW_INPUT_FILE = Path(TEST_BASE_DIR, "DHW_12x24x36_flow_dim_3x6x9_zero").resolve() # Resolves to "zero_flow_displacements.dat"
        AFFINE_MATRIX_FILE = Path(TEST_BASE_DIR, "identity_affine_matrix.txt").resolve()

        CPP_APPLY_BCV_OUTPUT_FILE = Path(TEST_OUTPUT_DIR, "./cpp_apply_bcv_label_output.nii.gz").resolve()

        case_args_cpp = [
            sys.argv[0], # Add the name of the calling programm
            '-M', str(MOVING_INPUT_FILE),
            '-O', str(FLOW_INPUT_FILE),
            '-A', str(AFFINE_MATRIX_FILE),
            '-D', str(CPP_APPLY_BCV_OUTPUT_FILE),
        ]

        TORCH_APPLY_BCV_OUTPUT_FILE = Path(TEST_BASE_DIR, "./test_output/cpp_apply_bcv_torch_label_output.nii.gz").resolve()

        case_args_torch = [
            '-M', str(MOVING_INPUT_FILE),
            '-O', str(FLOW_INPUT_FILE),
            '-A', str(AFFINE_MATRIX_FILE),
            '-D', str(TORCH_APPLY_BCV_OUTPUT_FILE),
        ]

        #########################################################
        # Write deeds output to harddrive
        CPP_APPLY_BCV_MODULE.applyBCV_main(len(case_args_cpp), case_args_cpp)

        #########################################################
        # Write torch output to harddrive
        apply_bcv_main(case_args_torch)

        #########################################################
        # Assert difference

        # Load files from disk again
        cpp_warped = torch.tensor(nib.load(CPP_APPLY_BCV_OUTPUT_FILE).get_fdata())
        torch_warped = torch.tensor(nib.load(TORCH_APPLY_BCV_OUTPUT_FILE).get_fdata())

        assert torch.allclose(cpp_warped, torch_warped,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"

    #test for more image datas
    def test_main_case_three(self):

        TEST_BASE_DIR = Path(TEST_DATA_DIR, "./case_3").resolve()
        OUTPUT_DIR = TEST_BASE_DIR.joinpath("test_output")
        OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
        MOVING_INPUT_FILE = Path(TEST_BASE_DIR, "label_moving.nii.gz").resolve()
        FLOW_INPUT_FILE = Path(TEST_BASE_DIR, "./deeds_bcv_output/case_3").resolve() # Resolves to "zero_flow_displacements.dat"
        AFFINE_MATRIX_FILE = Path(TEST_BASE_DIR, "./linear_bcv_output/linear_bcv_affine_matrix.txt").resolve()

        CPP_APPLY_BCV_OUTPUT_FILE = Path(OUTPUT_DIR, "cpp_apply_bcv_label_output.nii.gz").resolve()
        import numpy as np
        ZERO_FLOW_INPUT_FILE = "./zero_flow"
        float_array = np.zeros((180//4,140//4,190//4,3)).astype('float32')
        output_file = open(ZERO_FLOW_INPUT_FILE+"_displacements.dat", 'wb')
        float_array.tofile(output_file)
        output_file.close()
        FLOW_INPUT_FILE = ZERO_FLOW_INPUT_FILE
        case_args_cpp = [
            sys.argv[0], # Add the name of the calling programm
            '-M', str(MOVING_INPUT_FILE),
            '-O', str(FLOW_INPUT_FILE),
            '-A', str(AFFINE_MATRIX_FILE),
            '-D', str(CPP_APPLY_BCV_OUTPUT_FILE),
        ]

        TORCH_APPLY_BCV_OUTPUT_FILE = Path(OUTPUT_DIR, "./torch_apply_bcv_label_output.nii.gz").resolve()

        case_args_torch = [
            '-M', str(MOVING_INPUT_FILE),
            '-O', str(FLOW_INPUT_FILE),
            '-A', str(AFFINE_MATRIX_FILE),
            '-D', str(TORCH_APPLY_BCV_OUTPUT_FILE),
        ]

        #########################################################
        # Write deeds output to harddrive
        CPP_APPLY_BCV_MODULE.apply_bcv_main(len(case_args_cpp), case_args_cpp)

        #########################################################
        # Write torch output to harddrive
        apply_bcv_main(case_args_torch, CPP_APPLY_BCV_MODULE)

        #########################################################
        # Assert difference

        # Load files from disk again
        cpp_warped = torch.tensor(nib.load(CPP_APPLY_BCV_OUTPUT_FILE).get_fdata())
        torch_warped = torch.tensor(nib.load(TORCH_APPLY_BCV_OUTPUT_FILE).get_fdata())

        assert torch.allclose(cpp_warped, torch_warped,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"


    def test_main_case_four(self):

        TEST_BASE_DIR = Path(TEST_DATA_DIR, "./case_4").resolve()
        TEST_BASE_DIR.joinpath("test_output").mkdir(exist_ok=True, parents=True)
        MOVING_INPUT_FILE = Path(TEST_BASE_DIR, "label_moving_50_percent.nii.gz").resolve()
        FLOW_INPUT_FILE = Path(TEST_BASE_DIR, "./deeds_bcv_output/case_4").resolve() # Resolves to "zero_flow_displacements.dat"
        AFFINE_MATRIX_FILE = Path(TEST_BASE_DIR, "./linear_bcv_output/linear_bcv_affine_matrix.txt").resolve()

        CPP_APPLY_BCV_OUTPUT_FILE = Path(TEST_BASE_DIR, "./linear_bcv_output/cpp_apply_bcv_label_output.nii.gz").resolve()

        case_args_cpp = [
            sys.argv[0], # Add the name of the calling programm
            '-M', str(MOVING_INPUT_FILE),
            '-O', str(FLOW_INPUT_FILE),
            '-A', str(AFFINE_MATRIX_FILE),
            '-D', str(CPP_APPLY_BCV_OUTPUT_FILE),
        ]

        TORCH_APPLY_BCV_OUTPUT_FILE = Path(TEST_BASE_DIR, "./linear_bcv_output/cpp_apply_bcv_torch_label_output.nii.gz").resolve()

        case_args_torch = [
            '-M', str(MOVING_INPUT_FILE),
            '-O', str(FLOW_INPUT_FILE),
            '-A', str(AFFINE_MATRIX_FILE),
            '-D', str(TORCH_APPLY_BCV_OUTPUT_FILE),
        ]

        #########################################################
        # Write deeds output to harddrive
        CPP_APPLY_BCV_MODULE.applyBCV_main(len(case_args_cpp), case_args_cpp)

        #########################################################
        # Write torch output to harddrive
        apply_bcv_main(case_args_torch)

        #########################################################
        # Assert difference

        # Load files from disk again
        cpp_warped = torch.tensor(nib.load(CPP_APPLY_BCV_OUTPUT_FILE).get_fdata())
        torch_warped = torch.tensor(nib.load(TORCH_APPLY_BCV_OUTPUT_FILE).get_fdata())

        assert torch.allclose(cpp_warped, torch_warped,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



if __name__ == '__main__':
    # unittest.main()
    tests = TestApplyBCV()
    # tests.test_main_case_two()
    tests.test_main_case_three()
    # tests.test_main_case_four()