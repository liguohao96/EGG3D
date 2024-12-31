#include <torch/extension.h>
#include <vector>
#include <stdio.h>

namespace mesh2sd{
    std::vector<torch::Tensor> sd_query_forward_cuda(
        torch::Tensor faces,
        torch::Tensor query 
    );
    std::vector<torch::Tensor> sd_query_forward_cpu(
        torch::Tensor faces,
        torch::Tensor query 
    );

    std::vector<torch::Tensor> sd_query_forward(
        torch::Tensor faces,
        torch::Tensor query 
    ){
        switch(faces.device().type()){
            case c10::DeviceType::CUDA:
                return sd_query_forward_cuda(faces, query);
                break;
            case c10::DeviceType::CPU:
                return sd_query_forward_cpu(faces, query);
                break;
            default :
                TORCH_CHECK(0, "unsupport dispatch type ", faces.device().type());
        }
    }

    PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
        m.def("sd_query_forward", &sd_query_forward, "SigendDistance2Mesh forward");
    }
}