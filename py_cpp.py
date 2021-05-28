import torch
torch.ops.load_library("build/libd_sigmoid.dylib")
sig = torch.ops.deeds_cpp.d_sigmoid(torch.ones(1))
print(sig)