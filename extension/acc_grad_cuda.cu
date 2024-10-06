#include "acc_grad.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void acc_grad_opt::init(){
   init_base();
}

void acc_grad_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
}

void acc_grad_opt::reshape_top(at::TensorOptions options){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,width_});
    reshape_top_base(options,shapes);
}

template <typename scalar_t>
__global__ void acc_grad_forward_kernel(const int nthreads, const scalar_t* const bottom_data,
    scalar_t * const top_data, const int width, const scalar_t lambda) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int base = index*width;
        top_data[base] = -bottom_data[base] + lambda;
        base +=  1;
        for(int i = 1; i<width; i++,base++){
            top_data[base] = top_data[base-1] - bottom_data[base] + lambda;
        }
    }
}

std::vector<at::Tensor>  acc_grad_opt::forward_cuda(at::Tensor  bottom_data, float lambda) 
{
    reshape(bottom_data.size(0), 1, 1, bottom_data.size(1));
    reshape_top({bottom_data.options()});
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "acc_grad_forward_cuda", 
			([&] {
                    acc_grad_forward_kernel<< <CAFFE_GET_BLOCKS(num_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (num_,bottom_data.data_ptr<scalar_t>(),top_data_[0].data_ptr<scalar_t>(),width_,static_cast<scalar_t>(lambda));
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}
