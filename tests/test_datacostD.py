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



class TestdatacostD(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Load deeds_libtorch module
        deeds_libtorch_dir = Path(THIS_SCRIPT_DIR, "../deeds_libtorch")

        # Load datacostD.py
        datacostD_py_file = Path(deeds_libtorch_dir, "datacostD.py")
        self.datacostD = load_module_from_path(datacostD_py_file)

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


    def test_warpAffineS(self):

        #########################################################
        # Prepare inputs
        D, H, W =  2,2,2

   

        input_img=torch.ones((D,H,W),dtype=torch.int16)
        for i in range(W):
            if i%2==0:
                input_img[:,i,i+1]=0
    
        ## Generate some artificial displacements for x,y,z
        x_disp_field = torch.zeros(D,H,W)
        y_disp_field = torch.zeros(D,H,W)
        z_disp_field = torch.zeros(D,H,W)
        T=torch.eye(3,4)


         # u displacement



        #########################################################
        # Get deeds output
        print("\nRunning deeds 'warpAffineS': ")
        deeds_warped= self.applyBCV_module.applyBCV_warpAffineS(input_img,T,x_disp_field,y_disp_field,z_disp_field
            )



        #########################################################
        # Get torch output
        print("\nRunning torch 'warpAffineS': ")
        torch_warped = self.datacostD.warpAffineS(input_img,T,x_disp_field,y_disp_field,z_disp_field)



        #########################################################
        # Assert difference
        assert torch.allclose(deeds_warped, torch_warped,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"





if __name__ == '__main__':
    # unittest.main()
    tests = TestdatacostD()
    tests.test_warpAffineS()
