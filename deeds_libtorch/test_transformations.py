
import os
import torch
from torch.utils.cpp_extension import load

if __name__ == '__main__':

    factor = 1
    # input_u[0,0,0] = 2

    # input_v, input_w = input_u, input_u

    # _output_torch = torch.autograd.functional.jacobian(
    #     func, inputs, create_graph=False, strict=False, vectorize=False)

    # Expose interpolate3d function of transformation.h

    D, H, W = 3, 2, 1

    DELTA_W = +1.
    DELTA_H = +1.
    DELTA_D = +.5

    u_disp_field = torch.zeros(1,D,H,W,1)
    v_disp_field = torch.zeros(1,D,H,W,1)
    w_disp_field = torch.zeros(1,D,H,W,1)

    u_disp_field[0,:,1,:,0] = -2.0*(DELTA_W/W) # u displacement
    # v_disp_field[0,:,:,:,0] = -2.0*(DELTA_H/H) # v displacement
    # w_disp_field[0,:,:,:,0] = -2.0*(DELTA_D/D) # w displacement

    disp_field = torch.cat([u_disp_field, v_disp_field, w_disp_field], dim=-1)
    disp_field = disp_field.reshape(D,H,W,3)

    # def apply_displacement(_input, disp_field):
    #     _input = torch.rand(1,1,D,H,W)

    #     # disp_field_only_u = torch.zeros(1,D,H,W,3)
    #     # disp_field_only_u[:,:,:,:,0:1] = u_disp_field

    #     # disp_field_only_v = torch.zeros(1,D,H,W,3)
    #     # disp_field_only_v[:,:,:,:,1:2] = v_disp_field

    #     # disp_field_only_w = torch.zeros(1,D,H,W,3)
    #     # disp_field_only_w[:,:,:,:,2:3] = w_disp_field

    #     affine = torch.eye(3,4)
    #     affine = affine.unsqueeze(0)

    #     id_grid = torch.nn.functional.affine_grid(affine, size=(1, 1, D, H, W), align_corners=False)
    #     displaced = torch.nn.functional.grid_sample(_input, id_grid + disp_field.reshape(1,D,H,W,3), mode='bilinear')
    #     displaced = displaced.reshape(D,H,W)

    #     # displaced = (
    #     #     torch.nn.functional.grid_sample(_input, id_grid + disp_field_only_u, mode='bilinear'),
    #     #     torch.nn.functional.grid_sample(_input, id_grid + disp_field_only_v, mode='bilinear'),
    #     #     torch.nn.functional.grid_sample(_input, id_grid + disp_field_only_w, mode='bilinear')
    #     # )

    #     print("input\n", _input)
    #     print("output\n", displaced)
    #     return displaced.reshape(D,H,W)


    def jacobian(u_disp, v_disp, w_disp):

        WEIGHTS = torch.tensor([[-.5, 0., .5]])
        X_WEIGHTS = WEIGHTS.view(1, 1, 1, 3).repeat(1, 1, 1, 1, 1)
        Y_WEIGHTS = WEIGHTS.view(1, 1, 3, 1).repeat(1, 1, 1, 1, 1)
        Z_WEIGHTS = WEIGHTS.view(1, 3, 1, 1).repeat(1, 1, 1, 1, 1)

        def get_J_row_entries(disp):
            disp_field_envelope_x = torch.cat([disp[:,:,:,0:1,:], disp, disp[:,:,:,-1:,:]], dim=3)
            d_disp_over_dx = torch.nn.functional.conv3d(disp_field_envelope_x.reshape(1,1,D,H,W+2), X_WEIGHTS)

            disp_field_envelope_y = torch.cat([disp[:,:,0:1,:,:], disp, disp[:,:,-1:,:,:]], dim=2)
            d_disp_over_dy = torch.nn.functional.conv3d(disp_field_envelope_y.reshape(1,1,D,H+2,W), Y_WEIGHTS)

            disp_field_envelope_z = torch.cat([disp[:,0:1,:,:,:], disp, disp[:,-1:,:,:,:]], dim=1)
            d_disp_over_dz = torch.nn.functional.conv3d(disp_field_envelope_z.reshape(1,1,D+2,H,W), Z_WEIGHTS)

            return (d_disp_over_dx, d_disp_over_dy, d_disp_over_dz)

        J11, J12, J13  = get_J_row_entries(u_disp)
        J21, J22, J23  = get_J_row_entries(v_disp)
        J31, J32, J33  = get_J_row_entries(w_disp)

        return torch.stack([J11, J12, J13, J21, J22, J23, J31, J32, J33]).reshape(3, 3, D, H, W)

    jac_out  = 1/factor*jacobian(u_disp_field, v_disp_field, w_disp_field)
    jac_out[0,0]+=1
    jac_out[1,1]+=1
    jac_out[2,2]+=1

    J = torch.det(jac_out.reshape(-1,3,3))
    print("mean", J.mean(), "std", J.std())

    torch_jac_det_std = J.std()

    if os.environ.get('USE_JIT_COMPILE', None) == '1':
        # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
        applyBCV_module = load(name="applyBCV_module", sources=["src/applyBCV.cpp"], build_directory="./build-jit")
        jacobian_cpp = applyBCV_module.applyBCV_jacobian(
            u_disp_field.reshape(D,H,W),
            v_disp_field.reshape(D,H,W),
            w_disp_field.reshape(D,H,W),
            torch.tensor([factor], dtype=torch.int))

    else:
        # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
        torch.ops.load_library("build/liblibtorch-applyBCV.dylib")
        # jacobian_cpp = torch.ops.deeds_applyBCV.applyBCV_jacobian(_input_u, _input_w, _input_w, factor)

    assert torch.allclose(torch_jac_det_std, jacobian_cpp, rtol=1e-05, atol=1e-08, equal_nan=False), "Tensors do not match"
