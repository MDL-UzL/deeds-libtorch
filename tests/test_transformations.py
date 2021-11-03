import sys
import os
import unittest
import importlib.util
from pathlib import Path
import torch
from torch.utils.cpp_extension import load

os.environ['USE_JIT_COMPILE'] = '1'
THIS_SCRIPT_DIR = os.path.dirname(os.path.realpath(__file__))

def load_module_from_path(_path):
    # See https://stackoverflow.com/questions/67631/how-to-import-a-module-given-the-full-path?rq=1
    spec = importlib.util.spec_from_file_location(str(_path), _path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    return mod



class TestTransformations(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load deeds_libtorch module
        deeds_libtorch_dir = Path(THIS_SCRIPT_DIR, "../deeds_libtorch")

        # Load transformations.py
        transformations_py_file = Path(deeds_libtorch_dir, "transformations.py")
        self.transformations = load_module_from_path(transformations_py_file)

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



    def test_jacobian(self):

        #########################################################
        # Prepare inputs
        FACTOR = 1
        D, H, W =  6, 6, 2

        DELTA_W = +6.
        DELTA_H = +2.
        DELTA_D = +.5

        ## Generate some artificial displacements for x,y,z
        x_disp_field = torch.zeros(D,H,W)
        y_disp_field = torch.zeros(D,H,W)
        z_disp_field = torch.zeros(D,H,W)

        x_disp_field[0,0,0] = -2.0*(DELTA_W/W) # u displacement
        x_disp_field[0,0,1] = 2.0*(DELTA_W/W) # u displacement
        # x_disp_field[2,2,2] = -2.0*(DELTA_W/W) # u displacement
        # y_disp_field[:,:,:] = -2.0*(DELTA_H/H) # v displacement
        # z_disp_field[:,:,:] = -2.0*(DELTA_D/D) # w displacement



        #########################################################
        # Get deeds output
        print("\nRunning deeds 'jacobian': ")
        cpp_std_det_jac = self.applyBCV_module.applyBCV_jacobian(
            x_disp_field,
            y_disp_field,
            z_disp_field,
            torch.tensor([FACTOR], dtype=torch.int))



        #########################################################
        # Get torch output
        print("\nRunning torch 'std_det_jacobians': ")
        torch_std_det_jac = self.transformations.std_det_jacobians(
            x_disp_field, y_disp_field, z_disp_field, FACTOR
        )



        #########################################################
        # Assert difference
        assert torch.allclose(torch_std_det_jac, cpp_std_det_jac,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



    def test_interp3(self):

        #########################################################
        # Prepare inputs
        input_size = (2,2,2)
        _input = torch.zeros(input_size)
        # _input[0,0,0] = 0
        _input[0,0,0] = 1.

        print(_input.shape)

        output_size = (4,4,4)

        #########################################################
        # Get deeds output
        print("\nRunning deeds 'interp3': ")
        cpp_interp3 = self.applyBCV_module.applyBCV_interp3(_input, torch.Tensor(output_size), torch.tensor([False], dtype=torch.bool))
        print(cpp_interp3)

        #########################################################
        # Get torch output
        print("\nRunning torch 'interpolate': ")
        torch_interpolated = torch.nn.functional.interpolate(
            _input.unsqueeze(0).unsqueeze(0),
            output_size,
            mode='trilinear',
            align_corners=True
        ).squeeze(0).squeeze(0)
        print(torch_interpolated)

        #########################################################
        # Assert difference
        assert torch.allclose(torch_interpolated, cpp_interp3,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



if __name__ == '__main__':
    # unittest.main()
    tests = TestTransformations()
    tests.test_interp3()