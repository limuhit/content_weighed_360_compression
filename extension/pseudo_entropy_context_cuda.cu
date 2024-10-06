#include "pseudo_entropy_context.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>
#include "math_functions.hpp"

void pseudo_entropy_context_opt::init(){
    init_base();
}

__global__ void pseudo_entropy_context_init_kernel(const int nthreads, int * hindex, int * stride_inv,
    int width, int rt, float * weight) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        hindex[index] = static_cast<int>(weight[index]/64*width+0.5);
        stride_inv[index] = rt * width / hindex[index];
    }
}

bool pseudo_entropy_context_opt::reshape_hw(int num, int height, int width){
    height_ = height;
    width_ = width;
    num_ = num;
    //printf("pcontext width %d\n", width_);
    bool exists = (width_dict_.count(width)>0);
    bool update = (width_dict_update_.count(width)>0);
    //printf("count %d %d",width_dict_.count(width),width_dict_update_.count(width));
    if(exists && update) return false;
    //printf("pcontext new width %d\n", width_);
    if(!exists){
        width_dict_[width] = 1;
        h_out_ = height_ * npart_;
        w_out_ = width_;
        stride_inv_[width_] =  at::zeros({num_,npart_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
        hindex_[width_] =  at::zeros({num_,npart_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
    }
    int count = num_*npart_;
    pseudo_entropy_context_init_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
        (count, hindex_[width_].data_ptr<int>(), stride_inv_[width_].data_ptr<int>(), width_, rt_, weight_.data_ptr<float>());
    width_dict_update_[width] = 1;
    return true;
}

bool pseudo_entropy_context_opt::reshape_channel_pad(int channel, int pad){
    channel_ = channel; 
    pad_ = pad;
    //printf("pcontext channel %d pad %d\n", channel_,pad_);
    assert((channel<1000)&&(pad<10)&&"the channel number should be less than 1000 and the pad size should be less than 10");
    cp_ = width_*10000 + channel_*10 + pad_;
    bool exists = pad_channel_dict_.count(cp_)>0;
    bool update = pad_channel_dict_update_.count(cp_)>0;
    if(exists && update) return false;
    //printf("pcontext new channel %d pad %d\n", channel_,pad_);
    if(!exists){
        pad_channel_dict_[cp_] = 1;
        hindex2_[cp_] = at::zeros({num_, npart_, 2*pad_}, torch::TensorOptions().dtype(torch::kInt).device(torch::kCUDA, device_));
        param_[cp_] = at::zeros({num_, npart_, 2*pad_, width_, 4}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, device_));
        param2_[cp_] = at::zeros({num_, npart_, height_, width_*rt_}, torch::TensorOptions().dtype(torch::kFloat64).device(torch::kCUDA, device_));
    }
    return true;
}


__global__ void pseudo_entropy_context_forward_kernel_v0(const int nthreads, double * const param, const int * hindex, int * hindex2,
    const int channel, const int height, const int width, const int npart, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int hidx = index / width / pad / 2;
        int tg = hidx % npart;
        int tn = hidx / npart;
        if(tw>=hindex[hidx]) continue;
        int ph, pg;
        double pw;
        bool bound = false;
        if(tl==0){
            ph = tg*height - pad + tp;
            if(ph<0){
                bound = true;
            }else{
                pg = ph / height + tn * npart;
                pw = (tw + 0.5) / hindex[hidx] * hindex[pg] - 0.5 + 1e-9;
            }
        }else{
            ph = (tg+1)*height + tp;
            if(ph>=height*npart){
                bound = true;
            }else{
                pg = ph / height + tn * npart;
                pw = (tw + 0.5) / hindex[hidx] * hindex[pg] - 0.5 + 1e-9;
            }
        }
        
        if(bound){
            param[index*stride] =  (tl == 0) ? hidx *channel*(height+pad*2) + tp : hidx *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            if(tw==0){
                hindex2[((tn*npart + tg)*2 + tl)*pad + tp] = -1;
            }
        }else{
            param[index*stride] =  (tl == 0) ? hidx *channel*(height+pad*2) + tp : hidx *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            int pidx = pw<0? -1 : static_cast<int> (pw);
            
            param[index*stride + 2] = pidx;
            param[index*stride + 3] = pidx+1-pw;
            double qwa = (pidx + 1 + 0.5) / hindex[pg] * width - 0.5;
            double qwb = (tw + 0.5) / hindex[hidx] * width - 0.5;
            int qidx = static_cast<int> (qwb);
            //printf("%d %d %d %d %f %f %f\n",index, tg, tl, pidx, pw, qwa, qwb);
            if(qwa>=qidx+0.999){
                param[index*stride + 3] = 1.;
            }else{
                if(pidx==-1){
                    param[index*stride + 3] = 0.;
                }
            }
            param[index*stride+1] = (pg*channel*height + ph % height) * width; 
            if(tw==0){
                hindex2[((tn*npart + tg)*2 + tl)*pad + tp] = pg;
            }
        }
    }
}

__global__ void pseudo_entropy_context_forward_kernel_v1(const int nthreads, double * const param, const int * hindex, int * hindex2,
    const int channel, const int height, const int width, const int npart, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int hidx = index / width / pad / 2;
        int tg = hidx % npart;
        int tn = hidx / npart;
        if(tw>=hindex[hidx]) continue;
        int ph, pg;
        double pw;
        bool bound = false;
        if(tl==0){
            ph = tg*height - pad + tp;
            if(ph<0){
                bound = true;
            }else{
                pg = ph / height + tn * npart;
                pw = (tw + 0.5) / hindex[hidx] * hindex[pg] - 0.5 + 1e-9;
            }
        }else{
            ph = (tg+1)*height + tp;
            if(ph>=height*npart){
                bound = true;
            }else{
                pg = ph / height + tn * npart;
                pw = (tw + 0.5) / hindex[hidx] * hindex[pg] - 0.5 + 1e-9;
            }
        }
        
        if(bound){
            param[index*stride] =  (tl == 0) ? hidx *channel*(height+pad*2) + tp : hidx *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            if(tw==0){
                hindex2[((tn*npart + tg)*2 + tl)*pad + tp] = -1;
            }
        }else{
            param[index*stride] =  (tl == 0) ? hidx *channel*(height+pad*2) + tp : hidx *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            param[index*stride+1] = (pg*channel*height + ph % height) * width;
            int pidx = pw<0? -1 : static_cast<int> (pw);
            //printf("%d %d %d %d %f %d\n",index, tg, tl, pidx, pw, tw);
            if(pidx>tw){
                param[index*stride + 2] = -1;
                param[index*stride + 3] = 1.;
            }else if(pidx+1>tw){
                param[index*stride + 2] = pidx;
                param[index*stride + 3] = 1.;
            }else{
                param[index*stride + 2] = pidx;
                param[index*stride + 3] = pidx+1-pw;
                if(pidx==-1){
                    param[index*stride + 3] = 0.;
                }
            }
            if(tw==0){
                hindex2[((tn*npart + tg)*2 + tl)*pad + tp] = pg;
            }
        }
    }
}

__global__ void pseudo_entropy_context_forward_kernel_v2(const int nthreads, double * const param, const int * hindex, int * hindex2,
    const int channel, const int height, const int width, const int npart, const int pad, const int stride) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int hidx = index / width / pad / 2;
        int tg = hidx % npart;
        int tn = hidx / npart;
        if(tw>=hindex[hidx]) continue;
        int ph, pg;
        double pw;
        bool bound = false;
        if(tl==0){
            ph = tg*height - pad + tp;
            if(ph<0){
                bound = true;
            }else{
                pg = ph / height + tn * npart;
                pw = (tw + 0.5) / hindex[hidx] * hindex[pg] - 0.5 + 1e-9;
            }
        }else{
            ph = (tg+1)*height + tp;
            if(ph>=height*npart){
                bound = true;
            }else{
                pg = ph / height + tn * npart;
                pw = (tw + 0.5) / hindex[hidx] * hindex[pg] - 0.5 + 1e-9;
            }
        }
        
        if(bound){
            param[index*stride] =  (tl == 0) ? hidx *channel*(height+pad*2) + tp : hidx *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            if(tw==0){
                hindex2[((tn*npart + tg)*2 + tl)*pad + tp] = -1;
            }
        }else{
            int pidx = pw<0? -1 : static_cast<int> (pw);
            param[index*stride] =  (tl == 0) ? hidx *channel*(height+pad*2) + tp : hidx *channel*(height+pad*2) + pad + height + tp;
            param[index*stride] *= (width+pad*2);
            param[index*stride+1] = (pg*channel*height + ph % height) * width ;
            if(pidx>0){
                param[index*stride + 2] = pidx;
                param[index*stride + 3] = pidx+1-pw;
            }else{
                param[index*stride + 2] = -1;
                param[index*stride + 3] = 1.;
            }
            if(tw==0){
                hindex2[((tn*npart + tg)*2 + tl)*pad + tp] = pg;
            }
        }
    }
}

__global__ void pseudo_entropy_context_backward_kernel(const int nthreads, double * const inv_param, double * const param, const int * hindex,
    const int * hindex2, const int channel, const int height, const int width, const int npart, const int pad, const int stride, const int stride_out,
    const int * stride_inv) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int tw = index % width;
        int tp = (index / width) % pad;
        int tl = (index / width) / pad % 2;
        int tg = index / width / pad / 2;
        if(tw>=hindex[tg]) continue;
        if(hindex2[index/width]<0) continue;
        int pbase = static_cast<int>(param[index*stride+1]+1e-6);
        int pw = param[index*stride+2]<0? -1 : static_cast<int>(param[index*stride+2]+1e-6);
        int sbase = static_cast<int>(param[index*stride]+1e-6);
        double t = param[index*stride+3];
        int qh = pbase / width % height;
        int qg = pbase / width / height / channel;
        int qbase = (qg*height+qh)*stride_out;
        int idx, ti;
        if((pw>=0) && (t>0)){
            idx = qbase + pw*stride_inv[qg];
            //printf("%d %d %d %d\n", qg, qh, qbase, idx);
            ti = atomicAdd(inv_param + idx,  1.); 
            if(ti*2+4>stride_inv[qg]) printf("%d %d overflow!\n", tg, ti);
            if(ti==0)  inv_param[idx+1] = pbase + pw;
            inv_param[idx+ti*2+2] = sbase + tw + pad;
            inv_param[idx+ti*2+3] = t;
        }
        int pww = (pw+1)%hindex[qg];
        if(t<1){
            idx = qbase + pww*stride_inv[qg];
            ti = atomicAdd(inv_param + idx,  1.); 
            if(ti*2+4>stride_inv[qg]) printf("%d %d overflow!\n", tg, ti);
            if(ti==0) inv_param[idx+1] = pbase + pww;
            inv_param[idx+ti*2+2] = sbase + tw + pad;
            inv_param[idx+ti*2+3] = 1-t;
        }
        
    }
}

std::vector<at::Tensor> pseudo_entropy_context_opt::produce_param(int num, int channel, int height, int width, int pad)
{
    reshape_hw(num / npart_, height, width);
    bool change_flag = reshape_channel_pad(channel,pad);
	int count;
	if(change_flag){
        count = num_ * npart_  * width_ * pad_ * 2;
        caffe_gpu_set(stream_,count*4,double(0),param_[cp_].data_ptr<double>());
        caffe_gpu_set(stream_,num_*npart_*height_*width_*rt_,double(0),param2_[cp_].data_ptr<double>());
        switch(context_version_){
            case 0:
                pseudo_entropy_context_forward_kernel_v0<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, param_[cp_].data_ptr<double>(), hindex_[width_].data_ptr<int>(), hindex2_[cp_].data_ptr<int>(),
                        channel_, height_, width_, npart_, pad_, 4);
                break;
            case 1:
                pseudo_entropy_context_forward_kernel_v1<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, param_[cp_].data_ptr<double>(), hindex_[width_].data_ptr<int>(), hindex2_[cp_].data_ptr<int>(),
                        channel_, height_, width_, npart_, pad_, 4);
                break;
            case 2:
                pseudo_entropy_context_forward_kernel_v2<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                    (count, param_[cp_].data_ptr<double>(), hindex_[width_].data_ptr<int>(), hindex2_[cp_].data_ptr<int>(),
                        channel_, height_, width_, npart_, pad_, 4);
                break;
            default:
                printf("undefined context version\n");
                break;
        }
        pseudo_entropy_context_backward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
            (count, param2_[cp_].data_ptr<double>(), param_[cp_].data_ptr<double>(), hindex_[width_].data_ptr<int>(),hindex2_[cp_].data_ptr<int>(),
                channel_, height_, width_, npart_, pad_, 4, rt_*width_, stride_inv_[width_].data_ptr<int>());
        CUDA_POST_KERNEL_CHECK;
        pad_channel_dict_update_[cp_] = 1;
    }
    return {param_[cp_], param2_[cp_], hindex_[width_], hindex2_[cp_], stride_inv_[width_]};
}

at::Tensor pseudo_entropy_context_opt::produce_param_fill(int num, int height, int width) 
{
    reshape_hw(num/npart_,height,width);
    return hindex_[width_];
}