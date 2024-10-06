#include "sphere_uslice.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void sphere_uslice_opt::init(){
    init_base();
}

template <typename scalar_t>
__global__ void init_uslice_param_kernel(const int nthreads, const int npart, const int width, const int * hindex, scalar_t * param){
    CUDA_KERNEL_LOOP(index, nthreads) {
        int ti = index % width;
        int tp = index / width;
        int tw = hindex[tp];
        scalar_t nidx = (ti + 0.5) / width * tw - 0.5 + 1e-9;
        nidx = (nidx<0) ?  nidx + tw : nidx;
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

template <typename scalar_t>
__global__ void pseudo_uslice_init_kernel(const int nthreads, int * hindex, int width, scalar_t * weight) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        hindex[index] = static_cast<int>(weight[index]/64*width+0.5);
    }
}


void sphere_uslice_opt::reshape(int num, int channel, int height, int width){
    bool hflag = (num_==num);
    if (!reshape_base(num, channel, height-pad_*2, width-pad_*2)) return; 
    n_out_ = num_ / npart_;
    w_out_ = width_;
    h_out_ = height_ * npart_;
    if(hflag) return;
    hindex_ = at::zeros({n_out_, npart_}, torch::TensorOptions().dtype(torch::kInt32).device(torch::kCUDA, device_));
    init_param_ = true;
}

void sphere_uslice_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({n_out_,channel_, h_out_, w_out_});
    reshape_top_base(option,shapes);
    if(init_param_) resize_param_ = torch::zeros({n_out_,npart_,width_,5},option);
    init_param_ = false;
}

void sphere_uslice_opt::reshape_bottom(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_,channel_,height_+2*pad_,width_+2*pad_});
    reshape_bottom_base(option,shapes);
    //if(init_inv_) inv_param_ = at::zeros({npart_,width_*rt_}, option);
}


template <typename scalar_t>
__global__ void sphere_uslice_forward_kernel(const int nthreads, const scalar_t* const input,  
    scalar_t * const output, const scalar_t * param, const int * hindex, const int width, 
    const int height, const int height_out, const int channel, const int npart, 
    const int pad, const int stride_h, const int stride_w) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int th = (index / width) % height_out;
       int tc = (index / width / height_out) % channel;
       int tn = index / width / height_out / channel;
       int ph = th % height;
       int pb = th / height;
       int pn = tn * npart + pb;
       int base = (pn*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int pidx = ((pn*channel + tc)*stride_h + ph + pad) * stride_w + pad;
       int wl = hindex[pn];
       if(pw>0 && pw < wl-2){
           output[index] = param[base+1]*input[pidx+pw-1] + param[base+2]*input[pidx+pw] +
                           param[base+3]*input[pidx+pw+1] + param[base+4]*input[pidx+pw+2];
            //printf("%d %d %d %d %d\n", pb, pw-1, pw, pw+1,pw+2);
       }else{
           output[index] = param[base+1]*input[pidx+(pw-1+wl)%wl] + param[base+2]*input[pidx+pw] +
                           param[base+3]*input[pidx+(pw+1)%wl] + param[base+4]*input[pidx+(pw+2)%wl];
            //printf("%d %d %d %d %d\n", pb, (pw-1+wl)%wl, pw, (pw+1)%wl,(pw+2)%wl);
       }
   }
}

std::vector<at::Tensor>  sphere_uslice_opt::forward_cuda(at::Tensor  bottom_data, at::Tensor weight) 
{
    reshape(bottom_data.size(0), bottom_data.size(1), bottom_data.size(2), bottom_data.size(3));
    reshape_top(bottom_data.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		bottom_data.scalar_type(), "sphere_uslice_forward_cuda", 
			([&] {
                    count = n_out_ * npart_;
                    pseudo_uslice_init_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count,  hindex_.data_ptr<int>(), width_, weight.data_ptr<scalar_t>());
                    count = n_out_ * width_ * npart_;
                    init_uslice_param_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, npart_, width_, hindex_.data_ptr<int>(), resize_param_.data_ptr<scalar_t>());
                    count = n_out_ * channel_ * w_out_ * h_out_;
                    sphere_uslice_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_data.data_ptr<scalar_t>(), top_data_[0].data_ptr<scalar_t>(), resize_param_.data_ptr<scalar_t>(), 
                        hindex_.data_ptr<int>(), width_, height_, h_out_, channel_, npart_, pad_, height_+2*pad_, width_+2*pad_);
                    CUDA_POST_KERNEL_CHECK;

   			    }
			)
    );
    return top_data_;
}

template <typename scalar_t>
__global__ void sphere_uslice_init_inv_kernel(const int nthreads, 
    scalar_t * const inv_param, const scalar_t * param, const int * hindex, const int width, 
    const int npart, const int * stride_inv, const int stride) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int pb = (index / width) % npart;
       int base = (pb*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int wl = hindex[pb];
       scalar_t nidx;
       if(pw>0 && pw < wl-2){
           for (int j = -1; j<3; j++){
                nidx = atomicAdd(inv_param+pb*stride+stride_inv[pb]*(pw+j),1.);
                if(nidx*2+2>=stride_inv[pb]) printf("inv_param stack overflow! %d %d %d %d %d\n", pb, stride_inv[pb],tw,pw,j);
                inv_param[pb*stride+stride_inv[pb]*(pw+j)+static_cast<int>(nidx)*2+1] = tw;
                inv_param[pb*stride+stride_inv[pb]*(pw+j)+static_cast<int>(nidx)*2+2] = param[base+j+2];
           }
       }else{
            for (int j = -1; j<3; j++){
                nidx = atomicAdd(inv_param+pb*stride+stride_inv[pb]*((pw+j+wl)%wl),1.);
                if(nidx*2+2>=stride_inv[pb]) printf("inv_param stack overflow! %d %d %d %d %d\n", pb, stride_inv[pb],tw,pw,j);
                inv_param[pb*stride+stride_inv[pb]*((pw+j+wl)%wl)+static_cast<int>(nidx)*2+1] = tw;
                inv_param[pb*stride+stride_inv[pb]*((pw+j+wl)%wl)+static_cast<int>(nidx)*2+2] = param[base+j+2];
            }
       }
   }
}
/*
template <typename scalar_t>
__global__ void sphere_uslice_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const int * hindex, const int height, 
    const int height_out, const int channel, const int width, const int npart, 
    const int * stride_inv, const int stride, const  scalar_t * const inv_param, 
    const int pad, const int stride_h, const int stride_w) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int th = (index / width) % height;
        int tc = (index / width / height) % channel;
        int tn = index / width / height / channel;
        int tp = tn % npart;
        int pidx = ((tn * channel + tc) * stride_h + th + pad)*stride_w + tw + pad;
        //input[pidx] = 0;
        if(tw>=hindex[tp]) continue;
        int pn = tn / npart;
        int base = ((pn*channel + tc)*height_out + tp*height + th)*width;
        int wbase = tp * stride + tw*stride_inv[tp];
        int num = static_cast<int>(inv_param[wbase]+1e-6);
        for(int i=0;i<num;i++){
            input[pidx] = input[pidx] + output[base+static_cast<int>(inv_param[wbase+i*2+1])]*inv_param[wbase+i*2+2];
        }
    }    
}

std::vector<at::Tensor>  sphere_uslice_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_uslice_backward_cuda", 
			([&] {
                    if(init_inv_){
                        count = npart_ * width_;
                        caffe_gpu_set(stream_, count*rt_, 0, inv_param_.data_ptr<scalar_t>());
                        sphere_uslice_init_inv_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                            (count, inv_param_.data_ptr<scalar_t>(), resize_param_.data_ptr<scalar_t>(), hindex_.data_ptr<int>(), width_, 
                               npart_, stride_inv_.data_ptr<int>(), width_*rt_);
                        init_inv_ = false;
                    }
                    count = num_*channel_*height_*width_;
                    caffe_gpu_set(stream_, num_*channel_*(height_+2*pad_)*(width_+2*pad_),0, bottom_diff_[0].data_ptr<scalar_t>());
                    sphere_uslice_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), hindex_.data_ptr<int>(), height_, 
                        h_out_, channel_, width_, npart_,   stride_inv_.data_ptr<int>(), width_*rt_, inv_param_.data_ptr<scalar_t>(), 
                        pad_, height_+2*pad_, width_+2*pad_);
   			    }
			)
    );
    return {bottom_diff_[0],inv_param_,stride_inv_};
}
*/

template <typename scalar_t>
__global__ void sphere_uslice_backward_kernel(const int nthreads, scalar_t* const input,  
    const scalar_t * const output, const scalar_t * param, const int * hindex, const int width, 
    const int height, const int height_out, const int channel, const int npart, 
    const int pad, const int stride_h, const int stride_w) {
   CUDA_KERNEL_LOOP(index, nthreads) {
       int tw = index % width;
       int th = (index / width) % height_out;
       int tc = (index / width / height_out) % channel;
       int tn = index / width / height_out / channel;
       int ph = th % height;
       int pb = th / height;
       int pn = tn * npart + pb;
       int base = (pn*width+tw)*5;
       int pw = static_cast<int>(param[base]);
       int pidx = ((pn*channel + tc)*stride_h + ph + pad) * stride_w + pad;
       int wl = hindex[pn];
       if(pw>0 && pw < wl-2){
          // output[index] = param[base+1]*input[pidx+pw-1] + param[base+2]*input[pidx+pw] +
          //                 param[base+3]*input[pidx+pw+1] + param[base+4]*input[pidx+pw+2];
            atomicAdd(input+pidx+pw-1, output[index]*param[base+1]);
            atomicAdd(input+pidx+pw, output[index]*param[base+2]);
            atomicAdd(input+pidx+pw+1, output[index]*param[base+3]);
            atomicAdd(input+pidx+pw+2, output[index]*param[base+4]);
            //printf("%d %d %d %d %d\n", pb, pw-1, pw, pw+1,pw+2);
       }else{
            //output[index] = param[base+1]*input[pidx+(pw-1+wl)%wl] + param[base+2]*input[pidx+pw] +
            //               param[base+3]*input[pidx+(pw+1)%wl] + param[base+4]*input[pidx+(pw+2)%wl];
            atomicAdd(input+pidx+(pw-1+wl)%wl, output[index]*param[base+1]);
            atomicAdd(input+pidx+pw, output[index]*param[base+2]);
            atomicAdd(input+pidx+(pw+1)%wl, output[index]*param[base+3]);
            atomicAdd(input+pidx+(pw+2)%wl, output[index]*param[base+4]);
            //printf("%d %d %d %d %d\n", pb, (pw-1+wl)%wl, pw, (pw+1)%wl,(pw+2)%wl);
       }
   }
}

std::vector<at::Tensor>  sphere_uslice_opt::backward_cuda(at::Tensor  top_diff) 
{
    reshape_bottom(top_diff.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		top_diff.scalar_type(), "sphere_uslice_backward_cuda", 
			([&] {
                    caffe_gpu_set(stream_, num_*channel_*(height_+2*pad_)*(width_+2*pad_),0, bottom_diff_[0].data_ptr<scalar_t>());
                    count = n_out_ * channel_ * w_out_ * h_out_;
                    sphere_uslice_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, bottom_diff_[0].data_ptr<scalar_t>(), top_diff.data_ptr<scalar_t>(), resize_param_.data_ptr<scalar_t>(), 
                        hindex_.data_ptr<int>(), width_, height_, h_out_, channel_, npart_, pad_, height_+2*pad_, width_+2*pad_);
   			    }
			)
    );
    return {bottom_diff_[0]};
}