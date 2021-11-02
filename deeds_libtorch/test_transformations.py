
import os
import torch
from torch.utils.cpp_extension import load

if __name__ == '__main__':

    factor = 1

    D, H, W = 3, 2, 1

    DELTA_W = +1.
    DELTA_H = +1.
    DELTA_D = +.5

    u_disp_field = torch.zeros(1,D,H,W,1)
    v_disp_field = torch.zeros(1,D,H,W,1)
    w_disp_field = torch.zeros(1,D,H,W,1)

    u_disp_field[0,:,:,:,0] = -2.0*(DELTA_W/W) # u displacement
    v_disp_field[0,:,:,:,0] = -2.0*(DELTA_H/H) # v displacement
    w_disp_field[0,:,:,:,0] = -2.0*(DELTA_D/D) # w displacement

    disp_field = torch.cat([u_disp_field, v_disp_field, w_disp_field], dim=-1)
    disp_field = disp_field.reshape(D,H,W,3)


    if os.environ.get('USE_JIT_COMPILE', None) == '1':
        # Use just in time compilation. For this the source needs to contain a 'PYBIND11_MODULE' definition
        applyBCV_module = load(name="applyBCV_module", sources=["src/applyBCV.cpp"])
        jacobian_cpp = applyBCV_module.applyBCV_jacobian(
            u_disp_field.reshape(D,H,W),
            v_disp_field.reshape(D,H,W),
            w_disp_field.reshape(D,H,W), factor)

    else:
        # Use a precompiled library. For this the source needs to contain a 'TORCH_LIBRARY' definition
        torch.ops.load_library("build/liblibtorch-applyBCV.dylib")
        # jacobian_cpp = torch.ops.deeds_applyBCV.applyBCV_jacobian(_input_u, _input_w, _input_w, factor)

    assert torch.allclose(jacobian_torch, jacobian_cpp, rtol=1e-05, atol=1e-08, equal_nan=False), "Tensors do not match"
