#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
class acc_grad_opt: public base_opt{
	public:
		acc_grad_opt(int device=0, bool timeit=false){
			base_opt_init(device,timeit);
		}
		~acc_grad_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data, float lambda);
};
