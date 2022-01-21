# This is no unit test but a function to run applyBCV from python
import os
import sys
import unittest
import importlib.util
from pathlib import Path
import torch
from torch.utils.cpp_extension import load
import nibabel as nib
import torch.nn.functional as F

os.environ['USE_JIT_COMPILE'] = '1'
THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def load_module_from_path(_path):
    # See https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?rq=1
    spec = importlib.util.spec_from_file_location(str(_path), _path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod



class TestApplyBCV(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load deeds_libtorch module
        deeds_libtorch_dir = Path(THIS_SCRIPT_DIR, "../deeds_libtorch")

        # Load datacostD.py
        apply_bcv_py_file = Path(deeds_libtorch_dir, "applyBCV.py")
        self.applyBCV = load_module_from_path(apply_bcv_py_file)

        # Load build output
        src_dir = Path(THIS_SCRIPT_DIR, "../src").resolve()
        build_dir = Path(THIS_SCRIPT_DIR, "../build").resolve()
        build_jit_dir = Path(THIS_SCRIPT_DIR, "../build-jit").resolve()

        build_jit_dir.mkdir(exist_ok=True)

        apply_bcv_source = Path.joinpath(src_dir, "applyBCV.cpp").resolve()
        apply_bcv_dylib = Path.joinpath(build_dir, "liblibtorch-applyBCV.dylib").resolve()

        if os.environ.get('USE_JIT_COMPILE', None) == '1':
            # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
            self.applyBCV_module = load(name="applyBCV_module", sources=[apply_bcv_source], build_directory=build_jit_dir)

        else:
            # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
            torch.ops.load_library(apply_bcv_dylib)
            self.applyBCV_module = torch.ops.deeds_applyBCV
    
   
 

    def test_main_case_two(self):

        TEST_BASE_DIR = Path(THIS_SCRIPT_DIR, "./test_data/case_2").resolve()
        TEST_BASE_DIR.joinpath("test_output").mkdir(exist_ok=True, parents=True)
        MOVING_INPUT_FILE = Path(TEST_BASE_DIR, "label_toy_nifti_random_DHW_12x24x36.nii.gz").resolve()
        FLOW_INPUT_FILE = Path(TEST_BASE_DIR, "DHW_12x24x36_flow_dim_3x6x9_zero").resolve() # Resolves to "zero_flow_displacements.dat"
        AFFINE_MATRIX_FILE = Path(TEST_BASE_DIR, "identity_affine_matrix.txt").resolve()

        CPP_APPLY_BCV_OUTPUT_FILE = Path(TEST_BASE_DIR, "./test_output/cpp_apply_bcv_label_output.nii.gz").resolve()

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
        self.applyBCV_module.applyBCV_main(len(case_args_cpp), case_args_cpp)

        #########################################################
        # Write torch output to harddrive
        self.applyBCV.main(case_args_torch)

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

        TEST_BASE_DIR = Path(THIS_SCRIPT_DIR, "./test_data/case_3").resolve()
        TEST_BASE_DIR.joinpath("linear_bcv_output").mkdir(exist_ok=True, parents=True)
        MOVING_INPUT_FILE = Path(TEST_BASE_DIR, "moving.nii.gz").resolve()
        FLOW_INPUT_FILE = Path(TEST_BASE_DIR, "case_3").resolve() # Resolves to "zero_flow_displacements.dat"
        AFFINE_MATRIX_FILE = Path(TEST_BASE_DIR, "linear_bcv_affine_matrix.txt").resolve()

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
        self.applyBCV_module.applyBCV_main(len(case_args_cpp), case_args_cpp)

        #########################################################
        # Write torch output to harddrive
        self.applyBCV.main(case_args_torch)

        #########################################################
        # Assert difference

        # Load files from disk again
        cpp_warped = torch.tensor(nib.load(CPP_APPLY_BCV_OUTPUT_FILE).get_fdata())
        torch_warped = torch.tensor(nib.load(TORCH_APPLY_BCV_OUTPUT_FILE).get_fdata())

        assert torch.allclose(cpp_warped, torch_warped,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"
    

    def test_main_case_four(self):

        TEST_BASE_DIR = Path(THIS_SCRIPT_DIR, "./test_data/case_4").resolve()
        TEST_BASE_DIR.joinpath("linear_bcv_output").mkdir(exist_ok=True, parents=True)
        MOVING_INPUT_FILE = Path(TEST_BASE_DIR, "label_moving_50_percent.nii.gz").resolve()
        FLOW_INPUT_FILE = Path(TEST_BASE_DIR, "case_4").resolve() # Resolves to "zero_flow_displacements.dat"
        AFFINE_MATRIX_FILE = Path(TEST_BASE_DIR, "linear_bcv_affine_matrix.txt").resolve()

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
        self.applyBCV_module.applyBCV_main(len(case_args_cpp), case_args_cpp)

        #########################################################
        # Write torch output to harddrive
        self.applyBCV.main(case_args_torch)

        #########################################################
        # Assert difference

        # Load files from disk again
        cpp_warped = torch.tensor(nib.load(CPP_APPLY_BCV_OUTPUT_FILE).get_fdata())
        torch_warped = torch.tensor(nib.load(TORCH_APPLY_BCV_OUTPUT_FILE).get_fdata())

        assert torch.allclose(cpp_warped, torch_warped,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"
    

  
if __name__ == '__main__':
    tests = TestApplyBCV()
    tests.test_main_case_two()
    #tests.test_main_case_three()
    #tests.test_main_case_four()