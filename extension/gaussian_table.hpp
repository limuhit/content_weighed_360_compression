#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class gaussian_table_opt: public base_opt{
	public:
		gaussian_table_opt(int nstep, int nelement,float bias, float total_region, float beta=1e-9, int device = 0, bool timeit=false){
			nstep_ = nstep;
			bias_ = bias;
			tn_ = nelement;
			total_region_ = total_region;
			beta_ = beta;
			base_opt_init(device,timeit);
		}
		~gaussian_table_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
        void set_param(int nstep, float bias);
		std::vector<at::Tensor>  forward_cuda(at::Tensor delta, at::Tensor mean);
		int nstep_,tn_;
		float bias_,beta_;
		float total_region_;
};
