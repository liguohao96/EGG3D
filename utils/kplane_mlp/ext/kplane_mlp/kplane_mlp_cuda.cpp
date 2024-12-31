#include <torch/extension.h>

#include <vector>

#include "common.h"

// CUDA forward declarations

namespace at::native {

  torch::Tensor kplane_forward_cuda(
  // void kplane_mlp_forward_cuda(torch::Tensor &output,

      const torch::Tensor &kplane_input,
      const torch::Tensor &grid,
      const torch::Tensor &skip,

      const int64_t interpolation_mode, 
      const int64_t padding_mode,
      const bool    align_corners,
      const int64_t feature_fusion
  );
  torch::Tensor kplane_mlp_forward_cuda(
  // void kplane_mlp_forward_cuda(torch::Tensor &output,

      const torch::Tensor &kplane_input,
      const torch::Tensor &mlp_input,
      const torch::Tensor &grid,
      const torch::Tensor &skip,

      const int64_t interpolation_mode, 
      const int64_t padding_mode,
      const bool    align_corners,
      const int64_t feature_fusion,

      const int64_t mlp_layers, 
      const int64_t mlp_dim_hidden, 
      const int64_t mlp_dim_output, 
      const int64_t mlp_activation
  );
  std::vector<torch::Tensor> kplane_backward_cuda(
  // void kplane_mlp_backward_cuda(torch::Tensor &grad_kpl, torch::Tensor &grad_mlp, torch::Tensor &grad_grid, torch::Tensor &grad_skip,

      const torch::Tensor &grad_output,

      const torch::Tensor &kplane_input,
      const torch::Tensor &grid,
      const torch::Tensor &skip,

      int64_t interpolation_mode, 
      int64_t padding_mode,
      bool    align_corners,
      int64_t feature_fusion,

      std::array<bool, 3> output_mask 
  );
  std::vector<torch::Tensor> kplane_mlp_backward_cuda(
  // void kplane_mlp_backward_cuda(torch::Tensor &grad_kpl, torch::Tensor &grad_mlp, torch::Tensor &grad_grid, torch::Tensor &grad_skip,

      const torch::Tensor &grad_output,

      const torch::Tensor &kplane_input,
      const torch::Tensor &mlp_input,
      const torch::Tensor &grid,
      const torch::Tensor &skip,

      int64_t interpolation_mode, 
      int64_t padding_mode,
      bool    align_corners,
      int64_t feature_fusion,

      int64_t mlp_layers, 
      int64_t mlp_dim_hidden, 
      int64_t mlp_dim_output, 
      int64_t mlp_activation,

      std::array<bool, 4> output_mask 
  );
}

torch::Tensor kplane_forward(
// void kplane_mlp_forward(torch::Tensor &output,

    const torch::Tensor &kplane_input,
    const torch::Tensor &grid,
    const torch::Tensor &skip,

    int64_t interpolation_mode, 
    int64_t padding_mode,
    bool    align_corners,
    int64_t feature_fusion
) {
  return at::native::kplane_forward_cuda(// output,
                                  kplane_input, 
                                  grid, skip, 
                                  
                                  interpolation_mode, 
                                  padding_mode,
                                  align_corners,
                                  feature_fusion
                                  );
}

torch::Tensor kplane_mlp_forward(
// void kplane_mlp_forward(torch::Tensor &output,

    const torch::Tensor &kplane_input,
    const torch::Tensor &mlp_input,
    const torch::Tensor &grid,
    const torch::Tensor &skip,

    int64_t interpolation_mode, 
    int64_t padding_mode,
    bool    align_corners,
    int64_t feature_fusion,

    int64_t mlp_layers, 
    int64_t mlp_dim_hidden, 
    int64_t mlp_dim_output, 
    int64_t mlp_activation
) {

  // auto func = ([&](){
  //   switch (mlp_activation){
  //     case (Activation::Softplus):
  //       return at::native::kplane_mlp_forward_cuda<Activation::Softplus>;
  //     case (Activation::ReLU):
  //       return at::native::kplane_mlp_forward_cuda<Activation::ReLU>;
  //   }
  // })();

  return at::native::kplane_mlp_forward_cuda(// output,
                                  kplane_input, mlp_input,
                                  grid, skip, 
                                  
                                  interpolation_mode, 
                                  padding_mode,
                                  align_corners,
                                  feature_fusion,

                                  mlp_layers, 
                                  mlp_dim_hidden, 
                                  mlp_dim_output, 
                                  mlp_activation);
}

std::vector<torch::Tensor> kplane_backward(
// void kplane_mlp_backward(torch::Tensor &grad_kpl, torch::Tensor &grad_mlp, torch::Tensor &grad_grid, torch::Tensor &grad_skip,

    const torch::Tensor &grad_output,

    const torch::Tensor &kplane_input,
    const torch::Tensor &grid,
    const torch::Tensor &skip,

    int64_t interpolation_mode, 
    int64_t padding_mode,
    bool    align_corners,
    int64_t feature_fusion,

    std::array<bool, 3> output_mask 
) {
  return at::native::kplane_backward_cuda(//grad_kpl, grad_mlp, grad_grid, grad_skip,
    
                                  grad_output, kplane_input,
                                  grid, skip,

                                  interpolation_mode, 
                                  padding_mode,
                                  align_corners,
                                  feature_fusion,

                                  output_mask);
}

std::vector<torch::Tensor> kplane_mlp_backward(
// void kplane_mlp_backward(torch::Tensor &grad_kpl, torch::Tensor &grad_mlp, torch::Tensor &grad_grid, torch::Tensor &grad_skip,

    const torch::Tensor &grad_output,

    const torch::Tensor &kplane_input,
    const torch::Tensor &mlp_input,
    const torch::Tensor &grid,
    const torch::Tensor &skip,

    int64_t interpolation_mode, 
    int64_t padding_mode,
    bool    align_corners,
    int64_t feature_fusion,

    int64_t mlp_layers, 
    int64_t mlp_dim_hidden, 
    int64_t mlp_dim_output, 
    int64_t mlp_activation,

    std::array<bool, 4> output_mask 
) {
  return at::native::kplane_mlp_backward_cuda(//grad_kpl, grad_mlp, grad_grid, grad_skip,
    
                                  grad_output, kplane_input, mlp_input,
                                  grid, skip,

                                  interpolation_mode, 
                                  padding_mode,
                                  align_corners,
                                  feature_fusion,

                                  mlp_layers, 
                                  mlp_dim_hidden, 
                                  mlp_dim_output, 
                                  mlp_activation,
                                  output_mask);
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("kplane_forward",      &kplane_forward,      "kplane forward" );
  m.def("kplane_backward",     &kplane_backward,     "kplane backward");

  m.def("kplane_mlp_forward",  &kplane_mlp_forward,  "kplane_mlp forward" );
  m.def("kplane_mlp_backward", &kplane_mlp_backward, "kplane_mlp backward");
}