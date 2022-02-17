import os
import unittest
import torch

import time
import timeit
import numpy as np
import nibabel as nib

from deeds_libtorch.transformations import interp3, std_det_jacobians, vol_filter, upsampleDeformationsCL, consistentMappingCL


from __init__ import SRC_DIR, BUILD_DIR, BUILD_JIT_DIR, CPP_APPLY_BCV_MODULE, test_equal_tensors, log_wrapper


class TestTransformations(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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

        #########################################################
        # Get cpp output
        print("\nRunning deeds 'jacobian': ")
        cpp_std_det_jac = CPP_APPLY_BCV_MODULE.transformations_jacobian(
            x_disp_field,
            y_disp_field,
            z_disp_field,
            torch.tensor([FACTOR], dtype=torch.int))

        #########################################################
        # Get torch output
        print("\nRunning torch 'std_det_jacobians': ")
        torch_std_det_jac = std_det_jacobians(
            x_disp_field, y_disp_field, z_disp_field, FACTOR
        )

        #########################################################
        # Assert difference
        assert test_equal_tensors(torch_std_det_jac, cpp_std_det_jac)

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

        #########################################################
        # Get cpp output
        print("\nRunning deeds 'consistentMappingCL': ")
        cpp_consistentMappingCL = CPP_APPLY_BCV_MODULE.applyBCV_consistentMappingCL(
            x_disp_field,
            y_disp_field,
            z_disp_field,
            torch.tensor([FACTOR], dtype=torch.int))

        #########################################################
        # Get torch output
        print("\nRunning torch 'std_det_jacobians': ")
        torch_consistentMappingCL = consistentMappingCL(
            x_disp_field, y_disp_field, z_disp_field,x2_disp_field,y2_disp_field,z2_disp_field, FACTOR
        )

        #########################################################
        # Assert difference
        assert test_equal_tensors(torch_consistentMappingCL, cpp_consistentMappingCL)

    def test_interp3(self):

        #########################################################
        # Prepare inputs
        input_size = (1,3,3)
        _input = torch.zeros(input_size)
        _input[0,0,0] = 1.
        _input[0,-1,-1] = 10

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

        flag = False
        #########################################################
        # Get cpp output
        print("\nRunning deeds 'interp3': ")
        cpp_interp3 = CPP_APPLY_BCV_MODULE.transformations_interp3(
            _input,
            x1, y1, z1,
            torch.Tensor(output_size),
            torch.tensor([flag], dtype=torch.bool))

        #########################################################
        # Get torch output
        print("\nRunning torch 'interp3': ")
        torch_interp3 = interp3(
            _input,
            x1, y1, z1,
            output_size,
            flag
        )

        #########################################################
        # Assert difference
        assert test_equal_tensors(torch_interp3, cpp_interp3)

    def test_interp3_flag_set(self):

        #########################################################
        # Prepare inputs
        input_size = (1,3,3)
        _input = torch.zeros(input_size)
        _input[0,0,0] = 1.
        _input[0,-1,-1] = 10

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
        # Get cpp output
        print("\nRunning deeds 'interp3': ")
        cpp_interp3 = CPP_APPLY_BCV_MODULE.transformations_interp3(
            _input,
            x1, y1, z1,
            torch.Tensor(output_size),
            torch.tensor([flag], dtype=torch.bool))

        #########################################################
        # Get torch output
        print("\nRunning torch 'interp3': ")
        torch_interp3 = interp3(
            _input,
            x1, y1, z1,
            output_size,
            flag
        )

        #########################################################
        # Assert difference
        assert test_equal_tensors(torch_interp3, cpp_interp3)

    def test_interp3_complex_tensors(self):

        #########################################################
        # Prepare inputs
        input_size = (2,3,3)
        _input = torch.zeros(input_size)
        _input[0,0,0] = 1.
        _input[0,-1,-1] = 10

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

        flag = False
        #########################################################
        # Get cpp output
        cpp_interp3 = log_wrapper(
            CPP_APPLY_BCV_MODULE.transformations_interp3,
            _input,
            x1, y1, z1,
            torch.Tensor(output_size),
            torch.tensor([flag], dtype=torch.bool)
        )

        #########################################################
        # Get torch output
        torch_interp3 = log_wrapper(
            interp3,
            _input,
            x1, y1, z1,
            output_size,
            flag
        )

        #########################################################
        # Assert difference
        assert test_equal_tensors(torch_interp3, cpp_interp3)


    def test_volfilter(self):
            #########################################################
        # Prepare inputs
        input_size = (3,3,3)
        _input = torch.randn(input_size)
        _input[0,0,0] = 5.
        sigma=0.5
        kernel_sz=2

        #########################################################
        # Get cpp output
        print("\nRunning deeds 'vol_filter': ")
        cpp_volfilter = CPP_APPLY_BCV_MODULE.transformations_volfilter(_input,torch.tensor([kernel_sz]), torch.tensor([sigma]))

        #########################################################
        # Get torch output
        print("\nRunning torch 'vol_filter': ")
        torch_volfilter = vol_filter(_input, kernel_sz, sigma)

        #########################################################
        # Assert difference
        assert test_equal_tensors(torch_volfilter, cpp_volfilter)

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

        #########################################################
        # Get cpp output
        print("\nRunning deeds 'consistentMappingCL': ")
        deeds_u, deeds_v, deeds_w, deeds_u2, deeds_v2, deeds_w2 = CPP_APPLY_BCV_MODULE.transformations_consistentMappingCL(
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
        torch_u, torch_v, torch_w, torch_u2, torch_v2, torch_w2 = consistentMappingCL(
            x_disp_field, y_disp_field, z_disp_field,x2_disp_field,y2_disp_field,z2_disp_field, FACTOR
        )

        #########################################################
        # Assert difference
        assert torch.allclose(torch_u, deeds_u,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"

    def test_upsampleDeformationsCL(self):

        #########################################################
        # Prepare inputs
        INPUT_SIZE = torch.Size((2,2,2))
        UPSAMPLED_SIZE =  torch.Size((4,4,4))

        DELTA_VAL = +5.

        ## Generate some artificial displacements for x,y,z, fullsize / upsampled
        SIZE_HELPER_FIELD = torch.zeros(UPSAMPLED_SIZE)

        ##Generate input flow field
        u_input_flow = torch.zeros(INPUT_SIZE)
        v_input_flow = torch.zeros(INPUT_SIZE)
        w_input_flow = torch.zeros(INPUT_SIZE)

        u_input_flow[0,0,0] = -2.0*(DELTA_VAL) # u displacement
        u_input_flow[0,0,1] = 2.0*(DELTA_VAL) # u displacement

        v_input_flow[0,0,0] = -3.0*(DELTA_VAL) # u displacement
        v_input_flow[0,0,1] = 2.0*(DELTA_VAL) # u displacement

        w_input_flow[1,0,0] = -3.0*(DELTA_VAL) # u displacement
        w_input_flow[0,0,1] = 5.0*(DELTA_VAL) # u displacement

        #########################################################
        # Get cpp output
        print("\Input for deeds 'upsampleDeformationsCL': u_input_flow")
        print(u_input_flow)

        #########################################################
        # Get cpp output
        print("\nRunning deeds 'upsampleDeformationsCL': cpp_upsampled_u")
        (cpp_upsampled_u,
         cpp_upsampled_v,
         cpp_upsampled_w) = CPP_APPLY_BCV_MODULE.transformations_upsampleDeformationsCL(
                SIZE_HELPER_FIELD, SIZE_HELPER_FIELD, SIZE_HELPER_FIELD,
                u_input_flow, v_input_flow, w_input_flow,
            )
        print(cpp_upsampled_u)

        #########################################################
        # Get torch output
        print("\nRunning torch 'upsampleDeformationsCL': torch_upsampled_u")
        torch_upsampled_u, torch_upsampled_v, torch_upsampled_w = \
            upsampleDeformationsCL(
                SIZE_HELPER_FIELD, SIZE_HELPER_FIELD, SIZE_HELPER_FIELD,
                u_input_flow, v_input_flow, w_input_flow,
                UPSAMPLED_SIZE
            )
        print(torch_upsampled_u)

        #########################################################
        # Assert difference
        assert torch.allclose(torch_upsampled_u, cpp_upsampled_u,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"

        assert torch.allclose(torch_upsampled_v, cpp_upsampled_v,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"

        assert torch.allclose(torch_upsampled_w, cpp_upsampled_w,
            rtol=1e-05, atol=1e-08, equal_nan=False
        ), "Tensors do not match"



if __name__ == '__main__':
    # unittest.main()
    tests = TestTransformations()
    # tests.test_jacobian()
    # tests.test_interp3()
    # tests.test_interp3_flag_set()
    # tests.test_interp3_complex_tensors()
    # tests.test_volfilter()
    # tests.test_consistentMappingCL()
    tests.test_upsampleDeformationsCL()
