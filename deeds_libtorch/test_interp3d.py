
import torch
#
_input = torch.randn((3,4,5))

print(_input)

_output_torch = torch.nn.functional.interpolate(_input, scale_factor=.5, mode='nearest', align_corners=None, recompute_scale_factor=None)

print(_output)

# Expose interpolate3d function of transformation.h

_output_cpp = your_exposed_function(_input, ...)

_output_torch == _output_cpp
assert torch.isclose(_output_torch, _output_cpp, rtol=1e-05, atol=1e-08, equal_nan=False), "Tensors do not match"