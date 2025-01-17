#include "gaussian_table.hpp"
#include <curand.h>
#include <stdio.h>
#include <math.h>
#include <float.h>

void gaussian_table_opt::init(){
    init_base();
}

void gaussian_table_opt::set_param(int nstep,float bias){
    nstep_ = nstep;
    bias_ = bias;
}

void gaussian_table_opt::reshape(int num, int channel, int height, int width){
    if (!reshape_base(num, channel, height, width)) return; 
}

void gaussian_table_opt::reshape_top(at::TensorOptions option){
    std::vector<std::vector<int64_t>> shapes;
    shapes.push_back({num_*height_*width_, nstep_+1});
    reshape_top_base(option,shapes);
}

template <typename scalar_t>
__global__ void gaussian_table_forward_kernel(const int nthreads, const scalar_t* const delta,  const scalar_t* const mean, 
    scalar_t * const output,  const int ntable, const scalar_t total, const scalar_t bias, const scalar_t s2, const scalar_t bound) {
    CUDA_KERNEL_LOOP(index, nthreads) {
        int pt = index % ntable;
        int pn = index / ntable;
        scalar_t dt = delta[pn]<=bound ? bound : delta[pn];
        if(pt==0){
            output[index] = 0;
        }else if(pt==ntable-1){
            output[index] = static_cast<int>(total);
        }else{
            scalar_t v = pt - 1 - bias + 0.5, ps=0;
            ps = 0.5+0.5*erf(s2*(v-mean[pn])/dt);
            output[index] = static_cast<int>(total*ps+0.5);
        }
    }
}


template <typename scalar_t>
__global__ void gaussian_table_check_kernel(const int count, scalar_t * const output, const int ngroup) {
	CUDA_KERNEL_LOOP(index, count) {
		scalar_t bias = 0;
		scalar_t mval = 0;
		int midx = 0;
		for (int i = 0; i < ngroup; i++) {
			if (output[index*(ngroup+1) + i +1] <= output[index*(ngroup+1) + i])
			{
				bias += 1;
			}
            output[index*(ngroup+1) + i +1] += bias;
			if (output[index*(ngroup+1) + i+1] - output[index*(ngroup+1) + i] > mval) {
					mval = output[index*(ngroup+1) + i + 1] - output[index*(ngroup+1) + i];
					midx = i;
			}
		}
		if (bias > 0) {
			for (int i = midx; i < ngroup; i++) {
				output[index*(ngroup+1) + i+1] -= bias;
			}
		}	
	}
}

std::vector<at::Tensor>  gaussian_table_opt::forward_cuda(at::Tensor delta, at::Tensor mean) 
{
    reshape(mean.size(0), mean.size(1), mean.size(2), mean.size(3));
    reshape_top(mean.options());
	int count;
	AT_DISPATCH_FLOATING_TYPES(
		mean.scalar_type(), "gaussian_table_forward_cuda", 
			([&] {
                    count = tn_*(nstep_+1);
                    scalar_t s2 = 1. / sqrt(2.0);
                    gaussian_table_forward_kernel<< <CAFFE_GET_BLOCKS(count), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (count, delta.data_ptr<scalar_t>(),  mean.data_ptr<scalar_t>(),  top_data_[0].data_ptr<scalar_t>(),
                         nstep_+1, scalar_t(total_region_), scalar_t(bias_), s2, scalar_t(beta_));
                    gaussian_table_check_kernel<< <CAFFE_GET_BLOCKS(tn_), CAFFE_CUDA_NUM_THREADS, 0, stream_ >> >
                        (tn_, top_data_[0].data_ptr<scalar_t>(), nstep_);
                    CUDA_POST_KERNEL_CHECK;
   			    }
			)
    );
    return top_data_;
}