#include "sphere_slice.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void sphere_slice_opt::init(){
    init_base();
}

template <typename scalar_t>
__global__ void init_slice_param_kernel(const int nthreads, const int npart, const int width, const int * hindex, scalar_t * param){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ti = index % width;
        int tp = index / width;
        //int tn = index / width / npart;
        int tw = hindex[tp];
        if(ti<tw){
            scalar_t nidx = (ti + 0.5) / tw * width - 0.5 + 1e-9;
            nidx = (nidx<0) ?  nidx+width : nidx;
            scalar_t nint = static_cast<scalar_t>(static_cast<int>(nidx));
            scalar_t t = nidx - nint;
            scalar_t t2 = t*t;
            scalar_t t3 = t*t2;
            param[index*5] = nint;
            param[index*5+1] = (-t+2*t2-t3)/2; 
            param[index*5+2] = (2-5*t2+3*t3)/2;
            param[index*5+3] = (t+4*t2-3*t3)/2;
            param[index*5+4] = (-t2+t3)/2;   
        }
    }
}

template <typename scalar_t>
__global__ void pseudo_slice_init_kernel(const int nthreads, int * hindex, int width, scalar_t * weight) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        hindex[index] = static_cast<int>(weight[index]/64*width+0.5);
    }
}


void sphere_slice_opt::reshape(int num, int channel, int height, int width){
    bool hflag = (num_==num);
    if (!reshape_base(num, channel, height, width)) return;
    n_out_ = num_ * npart_; 
    h_out_ = height_ / npart_;
    w_out_ = width_;
    if(hflag) return;
    hindex_ = at::zeros({num, npart_}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_));
    init_param_ = true;
}

void sphere_slice_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({n_out_, channel_, h_out_+2*pad_, w_out_+2*pad_});
    reshape_top_base(option,shapes);
    if(init_param_) resize_param_ = torch::zeros({num_, npart_ , width_, 5},option);
    init_param_ = false;
}

void sphere_slice_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_,width_});
    reshape_bottom_base(option,shapes);
}


template <typename scalar_t>
__global__ void sphere_slice_forward_kernel(const int nthreads, const scalar_t* const input,  
    scalar_t * const output, const scalar_t * param, const int * hindex, const int width, 
    const int height, const int height_in, const int channel, const int npart, const int pad, 
    const int stride_h, const int stride_w) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int th = (index / width) % height;
       int tc = (index / width / height) % channel;
       int tn = index / width / height / channel;
       int oidx = ((tn*channel + tc)*stride_h + th + pad)*stride_w + tw + pad;
       int pn = tn / npart;
       int pt = tn % npart;
       int ph = th + pt * height;
       if(tw>=hindex[tn]){
           output[oidx] = 0;
           continue;
       } 
       int base = (tn*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int pidx = ((pn*channel + tc)*height_in + ph) * width;
       
       if(pw>0 && pw < width-2){
           output[oidx] = param[base+1]*input[pidx+pw-1] + param[base+2]*input[pidx+pw] +
                           param[base+3]*input[pidx+pw+1] + param[base+4]*input[pidx+pw+2];
       }else{
           output[oidx] = param[base+1]*input[pidx+(pw-1+width)%width] + param[base+2]*input[pidx+pw] +
                           param[base+3]*input[pidx+(pw+1)%width] + param[base+4]*input[pidx+(pw+2)%width];
       }
   }
}


std::vector<at::Tensor>  sphere_slice_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "sphere_slice_forward_cuda", 
			([&] {
                count = num_ * npart_;
                pseudo_slice_init_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count,  hindex_.data_ptr<int>(), width_, weight.data_ptr<scalar_t>());
                count = num_ * width_ * npart_;
                init_slice_param_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, npart_, width_, hindex_.data_ptr<int>(), resize_param_.data_ptr<scalar_t>());
                count = n_out_ * channel_ * w_out_ * h_out_;
                sphere_slice_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(),resize_param_.data_ptr<scalar_t>(),
                        hindex_.data_ptr<int>(), width_, h_out_, height_, channel_, npart_, pad_, h_out_+2*pad_, width_+2*pad_);
                CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}


template <typename scalar_t>
__global__ void sphere_slice_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const scalar_t * param, const int * hindex, const int width, 
    const int height, const int height_in, const int channel, const int npart, const int pad, 
    const int stride_h, const int stride_w) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int th = (index / width) % height;
       int tc = (index / width / height) % channel;
       int tn = index / width / height / channel;
       int oidx = ((tn*channel + tc)*stride_h + th + pad)*stride_w + tw + pad;
       int pn = tn / npart;
       int pt = tn % npart;
       int pbase = pn * 2 * npart + pt;
       int ph =  th + pt * height;
       if(tw>=hindex[tn]){
           continue;
       } 
       int base = (tn*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int pidx = ((pn*channel + tc)*height_in + ph) * width;
       
       if(pw>0 && pw < width-2){
            atomicAdd(input+pidx+pw-1, output[oidx]*param[base+1]);
            atomicAdd(input+pidx+pw, output[oidx]*param[base+2]);
            atomicAdd(input+pidx+pw+1, output[oidx]*param[base+3]);
            atomicAdd(input+pidx+pw+2, output[oidx]*param[base+4]);
       }else{
            atomicAdd(input+pidx+(pw-1+width)%width, output[oidx]*param[base+1]);
            atomicAdd(input+pidx+pw, output[oidx]*param[base+2]);
            atomicAdd(input+pidx+(pw+1)%width, output[oidx]*param[base+3]);
            atomicAdd(input+pidx+(pw+2)%width, output[oidx]*param[base+4]);
       }
   }
}

std::vector<at::Tensor>  sphere_slice_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_slice_backward_cuda", 
			([&] {
                    count = n_out_ * channel_ * w_out_ * h_out_;
                    caffe_gpu_set(stream_, num_*channel_*height_*width_, 0, bottom_diff_[0].data_ptr<scalar_t>());
                    sphere_slice_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), resize_param_.data_ptr<scalar_t>(),
                            hindex_.data_ptr<int>(), width_, h_out_, height_, channel_, npart_, pad_, h_out_+2*pad_, width_+2*pad_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return bottom_diff_;
}