
import torch
import transformations
#
_input = torch.randn((1,1,3,4,5))
print(_input)
_output_torch = transformations.interp3d(_input)
print(_output_torch)

# Expose interpolate3d function of transformation.h

#_output_cpp = your_exposed_function(_input, ...)

_output_torch == _output_cpp
assert torch.allclose(_output_torch, _output_cpp, rtol=1e-05, atol=1e-08, equal_nan=False), "Tensors do not match"
