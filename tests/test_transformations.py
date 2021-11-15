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
        FACTOR = 0.5
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
        input_size = [2,2,2]
        _input = torch.zeros(input_size)
        # _input[0,0,0] = 0
        _input[0,0,0] = 1.
        # _input[1,0,0] = .5
        # _input[1,1,1] = 1.

        print(_input.shape)

        output_size = [4,4,4]

        #########################################################
        # Get deeds output
        print("\nRunning deeds 'interp3': ")
        cpp_interp3 = self.applyBCV_module.applyBCV_interp3(_input, torch.Tensor(output_size), torch.tensor([False], dtype=torch.bool))
        print(cpp_interp3)

        #########################################################
        # Get torch output
        print("\nRunning torch 'interpolate': ")
        torch_interpolated = self.transformations.interp3d(_input,output_size)
        print(torch_interpolated)

        print("\nRunning grid output: ")
        N, C, H_in, W_in, D_in = 1,1,2,2,2

        # Create identity grid
        affine = torch.eye(3,4)
        affine = affine.unsqueeze(0)
        id_grid_output_size = (1,1, *output_size)

        def pad_same_around(paddeee, pad_count_per_side):
            padded = torch.cat([paddeee[0:1,:,].repeat(pad_count_per_side,1,1), paddeee, paddeee[-1:,:,:].repeat(pad_count_per_side,1,1)], dim=0)
            padded = torch.cat([padded[:,0:1,:].repeat(1,pad_count_per_side,1), padded, padded[:,-1:,:].repeat(1,pad_count_per_side,1)], dim=1)
            padded = torch.cat([padded[:,:,0:1].repeat(1,1,pad_count_per_side), padded, padded[:,:,-1:].repeat(1,1,pad_count_per_side)], dim=2)
            return padded

        id_grid = torch.nn.functional.affine_grid(affine, size=id_grid_output_size, align_corners=False)
        grid_output = torch.nn.functional.grid_sample(pad_same_around(_input, 1).unsqueeze(0).unsqueeze(0), id_grid*1.2, mode='bilinear', align_corners=False)
        print(grid_output)
        #########################################################
        # Assert difference
        assert torch.allclose(torch_volfilter, cpp_volfilter,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



    def test_consistentMappingCL(self):

        #########################################################
        # Prepare inputs
        FACTOR = 1
        D, H, W =  6, 6, 2

        DELTA_W = +6.
        DELTA_H = +2.
        DELTA_D = +.5

        DELTA_W2 = +7.
        DELTA_H2 = +3.
        DELTA_D2 = +.6

        ## Generate some artificial displacements for x,y,z
        x_disp_field = torch.zeros(D,H,W)
        y_disp_field = torch.zeros(D,H,W)
        z_disp_field = torch.zeros(D,H,W)

        ##Generate 2nd flow field
        x2_disp_field = torch.zeros(D,H,W)
        y2_disp_field = torch.zeros(D,H,W)
        z2_disp_field = torch.zeros(D,H,W)

        x_disp_field[0,0,0] = -2.0*(DELTA_W/W) # u displacement
        x_disp_field[0,0,1] = 2.0*(DELTA_W/W) # u displacement
        x2_disp_field[0,0,0] = -2.0*(DELTA_W2/W) # u displacement
        x2_disp_field[0,0,1] = 2.0*(DELTA_W2/W) # u displacement
        # x_disp_field[2,2,2] = -2.0*(DELTA_W/W) # u displacement
        # y_disp_field[:,:,:] = -2.0*(DELTA_H/H) # v displacement
        # z_disp_field[:,:,:] = -2.0*(DELTA_D/D) # w displacement



        #########################################################
        # Get deeds output
        print("\nRunning deeds 'consistentMappingCL': ")
        cpp_consistentMappingCL = self.applyBCV_module.applyBCV_consistentMappingCL(
            x_disp_field,
            y_disp_field,
            z_disp_field,
            torch.tensor([FACTOR], dtype=torch.int))



        #########################################################
        # Get torch output
        print("\nRunning torch 'std_det_jacobians': ")
        torch_consistentMappingCL = self.transformations.consistentMappingCL(
            x_disp_field, y_disp_field, z_disp_field,x2_disp_field,y2_disp_field,z2_disp_field, FACTOR
        )



        #########################################################
        # Assert difference
        assert torch.allclose(torch_consistentMappingCL, cpp_consistentMappingCL,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



    def test_interp3(self):

        #########################################################
        # Prepare inputs
        input_size = (1,3,3)
        _input = torch.zeros(input_size)
        # _input[0,0,0] = 0
        _input[0,0,0] = 1.
        # _input[0,1,-1] = -7.
        _input[0,-1,-1] = 10
        # _input[0,1,2] = 10.

        print("'interp3' input:")
        print(_input)

        output_size = (4,6,3)

        scale_m, scale_n, scale_o = [out_s/in_s for out_s, in_s in zip(output_size, input_size)]

        x1 = torch.zeros(output_size)
        y1 = torch.zeros(output_size)
        z1 = torch.zeros(output_size)
        m, n, o = output_size
        for k in range(o):
            for j in range(n):
                for i in range(m):
                    x1[i,j,k]=i/scale_m; # x helper var -> stretching factor in x-dir (gridded_size/full_size) at every discrete x (full size)
                    y1[i,j,k]=j/scale_n; # y helper var
                    z1[i,j,k]=k/scale_o; # z helper var

        flag = True
        #########################################################
        # Get deeds output
        print("\nRunning deeds 'interp3': ")
        cpp_interp3 = self.applyBCV_module.applyBCV_interp3(
            _input,
            x1, y1, z1,
            torch.Tensor(output_size),
            torch.tensor([flag], dtype=torch.bool))
        print(cpp_interp3)

        cpp_interp3_not_flag = self.applyBCV_module.applyBCV_interp3(
            _input,
            x1, y1, z1,
            torch.Tensor(output_size),
            torch.tensor([not flag], dtype=torch.bool))

        print("\nDifference: ")
        print(cpp_interp3_not_flag)
        #########################################################
        # Get torch output
        print("\nRunning torch 'interp3': ")
        torch_interp3 = self.transformations.interp3(
            _input,
            x1, y1, z1,
            output_size,
            flag
        )

        print(torch_interp3)

        #########################################################
        # Assert difference
        assert torch.allclose(torch_interp3, cpp_interp3,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"


    def test_volfilter(self):
            #########################################################
        # Prepare inputs
        input_size = (3,3,3)
        _input = torch.randn(input_size)
        print("input is",_input)
        # _input[0,0,0] = 0
        _input[0,0,0] = 5.
        sigma=0.5
        kernel_sz=2


        print(_input.shape)

        #########################################################
        # Get deeds output
        print("\nRunning deeds 'vol_filter': ")
        cpp_volfilter = self.applyBCV_module.applyBCV_volfilter(_input,torch.tensor([kernel_sz]), torch.tensor([sigma]))
        print(cpp_volfilter)

        #########################################################
        # Get torch output
        print("\nRunning torch 'vol_filter': ")
        torch_volfilter = self.transformations.vol_filter(_input,kernel_sz,sigma)
        print("Vol_filter is:",torch_volfilter)
        #########################################################
        # Assert difference
        assert torch.allclose(torch_volfilter, cpp_volfilter,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



    def test_consistentMappingCL(self):

        #########################################################
        # Prepare inputs
        FACTOR = 1
        D, H, W =  6, 6, 2

        DELTA_W = +6.
        DELTA_H = +2.
        DELTA_D = +.5

        DELTA_W2 = +7.
        DELTA_H2 = +3.
        DELTA_D2 = +.6

        ## Generate some artificial displacements for x,y,z
        x_disp_field = torch.zeros(D,H,W)
        y_disp_field = torch.zeros(D,H,W)
        z_disp_field = torch.zeros(D,H,W)

        ##Generate 2nd flow field
        x2_disp_field = torch.zeros(D,H,W)
        y2_disp_field = torch.zeros(D,H,W)
        z2_disp_field = torch.zeros(D,H,W)

        x_disp_field[0,0,0] = -2.0*(DELTA_W/W) # u displacement
        x_disp_field[0,0,1] = 2.0*(DELTA_W/W) # u displacement
        x2_disp_field[0,0,0] = -2.0*(DELTA_W2/W) # u displacement
        x2_disp_field[0,0,1] = 2.0*(DELTA_W2/W) # u displacement
        # x_disp_field[2,2,2] = -2.0*(DELTA_W/W) # u displacement
        # y_disp_field[:,:,:] = -2.0*(DELTA_H/H) # v displacement
        # z_disp_field[:,:,:] = -2.0*(DELTA_D/D) # w displacement



        #########################################################
        # Get deeds output
        print("\nRunning deeds 'consistentMappingCL': ")
        cpp_consistentMappingCL = self.applyBCV_module.applyBCV_consistentMappingCL(
            x_disp_field,
            y_disp_field,
            z_disp_field,
            x2_disp_field,
            y2_disp_field,
            z2_disp_field,
            torch.tensor([FACTOR], dtype=torch.int))



        #########################################################
        # Get torch output
        print("\nRunning torch 'consistent mapping': ")
        torch_consistentMappingCL = self.transformations.consistentMappingCL(
            x_disp_field, y_disp_field, z_disp_field,x2_disp_field,y2_disp_field,z2_disp_field, FACTOR
        )



        #########################################################
        # Assert difference
        assert torch.allclose(torch_consistentMappingCL[0], cpp_consistentMappingCL[0],
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"





if __name__ == '__main__':
    # unittest.main()
    tests = TestTransformations()
    tests.test_consistentMappingCL()
    tests.test_interp3()
    # tests.test_volfilter()
    # tests.test_consistentMappingCL()
