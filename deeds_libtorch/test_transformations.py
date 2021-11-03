
import os
import torch
from torch.utils.cpp_extension import load

if __name__ == '__main__':

    factor = 1

    D, H, W =  6, 2, 2

    DELTA_W = +6.
    DELTA_H = +2.
    DELTA_D = +.5

    x_disp_field = torch.zeros(D,H,W)
    y_disp_field = torch.zeros(D,H,W)
    z_disp_field = torch.zeros(D,H,W)

    x_disp_field[0,0,0] = -2.0*(DELTA_W/W) # u displacement
    x_disp_field[0,0,1] = 2.0*(DELTA_W/W) # u displacement
    # x_disp_field[2,2,2] = -2.0*(DELTA_W/W) # u displacement, comment this line to provoke error
    # y_disp_field[:,:,:] = -2.0*(DELTA_H/H) # v displacement
    # z_disp_field[:,:,:] = -2.0*(DELTA_D/D) # w displacement

    # disp_field = torch.stack([x_disp_field, y_disp_field, z_disp_field], dim=-1)



    def jacobians(x_disp_field, y_disp_field, z_disp_field):
        # Returns n=DxHxW jacobian matrices (3x3) of displacements per dimension
        #
        # Jacobian for a single data point (x,y,z) is
        #
        #   d_x_disp_field/dx  d_x_disp_field/dy  d_x_disp_field/dz
        #   d_y_disp_field/dx  d_y_disp_field/dy  d_y_disp_field/dz
        #   d_z_disp_field/dx  d_z_disp_field/dy  d_z_disp_field/dz

        assert x_disp_field.shape == y_disp_field.shape == z_disp_field.shape, \
            "Displacement field sizes must match."

        D, H, W = x_disp_field.shape

        # Prepare convolutions
        WEIGHTS = torch.tensor([[-.5, 0., .5]])
        KERNEL_SIZE = WEIGHTS.numel()

        ## Reshape weights for x,y,z dimension
        X_WEIGHTS = WEIGHTS.view(1, 1, 1, KERNEL_SIZE).repeat(1, 1, 1, 1, 1)
        Y_WEIGHTS = WEIGHTS.view(1, 1, KERNEL_SIZE, 1).repeat(1, 1, 1, 1, 1)
        Z_WEIGHTS = WEIGHTS.view(1, KERNEL_SIZE, 1, 1).repeat(1, 1, 1, 1, 1)

        def jacobians_row_entries(disp):
            # Convolute d_disp over d_x - change of displacement per x dimension
            disp_field_envelope_x = torch.cat(
                [
                    disp[:,:,0:1],
                    disp,
                    disp[:,:,-1:]
                ],
                dim=2)
            d_disp_over_dx = torch.nn.functional.conv3d(
                disp_field_envelope_x.reshape(1,1,D,H,W+2), X_WEIGHTS
            ).reshape(D,H,W)

            # Convolute d_disp over d_y - change of displacement per y dimension
            disp_field_envelope_y = torch.cat(
                [
                    disp[:,0:1,:],
                    disp,
                    disp[:,-1:,:]
                ],
                dim=1)
            d_disp_over_dy = torch.nn.functional.conv3d(
                disp_field_envelope_y.reshape(1,1,D,H+2,W), Y_WEIGHTS
            ).reshape(D,H,W)

            # Convolute d_disp over d_z - change of displacement per z dimension
            disp_field_envelope_z = torch.cat(
                [
                    disp[0:1,:,:],
                    disp,
                    disp[-1:,:,:]
                ],
                dim=0)
            d_disp_over_dz = torch.nn.functional.conv3d(
                disp_field_envelope_z.reshape(1,1,D+2,H,W), Z_WEIGHTS
            ).reshape(D,H,W)

            return (d_disp_over_dx, d_disp_over_dy, d_disp_over_dz)

        J11, J12, J13  = jacobians_row_entries(x_disp_field) # First row
        J21, J22, J23  = jacobians_row_entries(y_disp_field) # Second row
        J31, J32, J33  = jacobians_row_entries(z_disp_field) # Third row

        _jacobians = torch.stack(
            [J11, J12, J13, J21, J22, J23, J31, J32, J33]
        ).reshape(3, 3, D, H, W)

        return _jacobians



    def std_det_jacobians(_jacobians, factor):
        # Calculate
        assert _jacobians.shape[0] == _jacobians.shape[1], "Jacobian needs to be square."

        jac_square_len = _jacobians.shape[0]
        spatial_len = int(_jacobians.numel()/(jac_square_len**2))

        _jacobians  = 1 / factor * _jacobians
        _jacobians[0,0]+=1
        _jacobians[1,1]+=1
        _jacobians[2,2]+=1

        jac_deteterminants = torch.det(
            _jacobians.reshape(jac_square_len,jac_square_len,spatial_len).permute(2,0,1)
        )
        std_det_jacobians = jac_deteterminants.std()

        # TODO Probably move print method elsewhere
        print(f"mean(J)={jac_deteterminants.mean()}", f"std(J)={jac_deteterminants.std()}", f"(J<0)={(jac_deteterminants<0).sum()/(D*H*W)*100:.5f}%")

        return std_det_jacobians

    torch_std_det_jac = std_det_jacobians(jacobians(x_disp_field, y_disp_field, z_disp_field), factor)

    if os.environ.get('USE_JIT_COMPILE', None) == '1':
        # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
        applyBCV_module = load(name="applyBCV_module", sources=["src/applyBCV.cpp"], build_directory="./build-jit")
        cpp_std_det_jac = applyBCV_module.applyBCV_jacobian(
            x_disp_field,
            y_disp_field,
            z_disp_field,
            torch.tensor([factor], dtype=torch.int))

    else:
        # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
        torch.ops.load_library("build/liblibtorch-applyBCV.dylib")
        # jacobian_cpp = torch.ops.deeds_applyBCV.applyBCV_jacobian(_input_u, _input_w, _input_w, factor)

    assert torch.allclose(torch_std_det_jac, cpp_std_det_jac, rtol=1e-05, atol=1e-08, equal_nan=False), "Tensors do not match"
