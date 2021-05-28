#include <torch/script.h>

#include <iostream>

torch::Tensor d_sigmoid(torch::Tensor z) {
  auto s = torch::sigmoid(z);
  return (1 - s) * s;
}

TORCH_LIBRARY(deeds_cpp, m) {
  m.def("d_sigmoid", d_sigmoid);
}