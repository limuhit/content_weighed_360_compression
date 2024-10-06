#include "pseudo_split.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void pseudo_split_opt::init(){
    init_base();
}

void pseudo_split_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
    nout_ = num * npart_;
    h_out_ = height_ / npart_;
    w_out_ = width_;
    switch(context_version_){
        case 0:
            hindex_ = pctx_->produce_param_fill(num,h_out_,width);
            break;
        case 1:
            hindex_ = pectx_->produce_param_fill(num, h_out_,width);
            break;
        default:
            hindex_ = ectx_->produce_param_fill(h_out_,width); 
            break;
    }
    //printf("hindex addr: %p\n", hindex_.data_ptr<int>());
}

void pseudo_split_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({nout_,channel_, h_out_, w_out_});
    reshape_top_base(option,shapes);
}

void pseudo_split_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}

template <typename scalar_t>
__global__ void pseudo_split_backward_kernel(const int nthreads,  scalar_t * const input, const scalar_t * const output,
    const int * hindex, const int width, const int height, const int channel, const int npart, const int h_out) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pc = (index / width / height) % channel;
        int pg = (index / width / height / channel) % npart;
        int pn = index / width / height / channel / npart;
        int pidx = pn * npart + pg;
        int pad = (width - hindex[pidx]) / 2;
        if(pw>=hindex[pidx]) continue;
        int tidx = ((pn * channel + pc) * h_out + pg*height + ph) * width + pad + pw;
        input[tidx] = output[index];
    }
}

template <typename scalar_t>
__global__ void pseudo_split_forward_kernel(const int nthreads, const scalar_t * const input, scalar_t * const output,
    const int * hindex, const int width, const int height, const int channel, const int npart, const int h_out) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pw = index % width;
        int ph = (index / width) % height;
        int pc = (index / width / height) % channel;
        int pg = (index / width / height / channel) % npart;
        int pn = index / width / height / channel / npart;
        int pidx = pn * npart + pg;
        int pad = (width - hindex[pidx]) / 2;
        if(pw>=hindex[pidx]) continue;
        int tidx = ((pn * channel + pc) * h_out + pg*height + ph) * width + pad + pw;
        output[index] = input[tidx]; 
    }
}

std::vector<at::Tensor>  pseudo_split_opt::forward_cuda(at::Tensor  bottom_data) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "pseudo_split_forward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    caffe_gpu_set(stream_, count, scalar_t(0), top_data_[0].data_ptr<scalar_t>());
                    pseudo_split_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), hindex_.data_ptr<int>(), 
                            width_, h_out_, channel_, npart_, height_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}


std::vector<at::Tensor>  pseudo_split_opt::backward_cuda(at::Tensor  top_diff) 
{
    int count;
    reshape_bottom(top_diff.options());
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "pseudo_split_backward_cuda", 
			([&] {
                    count = num_ * channel_ * width_ * height_;
                    caffe_gpu_set(stream_, count, scalar_t(0), bottom_diff_[0].data_ptr<scalar_t>());
                    pseudo_split_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), hindex_.data_ptr<int>(), 
                            width_, h_out_, channel_, npart_, height_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}