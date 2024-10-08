#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include "pseudo_context.hpp"
#include "pseudo_entropy_context.hpp"
#include "entropy_context.hpp"
#include "string2class.hpp"
#include <string>

class pseudo_merge_opt: public base_opt{
	public:
		pseudo_merge_opt(int npart, std::string addr, int context_version, int device = 0, bool timeit=false){
			npart_ = npart;
			context_version_ = context_version;
			switch (context_version_)
			{
				case 0:
					pctx_ = FromStringPseudo(addr);
					break;
				case 1:
					pectx_ = FromStringPseudoEntropy(addr);
					break;
				default:
					ectx_ = FromString(addr);
					break;
			}
			base_opt_init(device,timeit);
		}
		~pseudo_merge_opt(){}
		void init();
		void reshape(int num, int channel, int height, int width);
        void reshape_top(at::TensorOptions options);
		void reshape_bottom(at::TensorOptions options);
		std::vector<at::Tensor>  forward_cuda(at::Tensor  bottom_data);
		std::vector<at::Tensor>  backward_cuda(at::Tensor  top_diff);
		int npart_;
		pseudo_context_opt* pctx_;
		pseudo_entropy_context_opt* pectx_;
		entropy_context* ectx_;
		at::Tensor hindex_;
		int context_version_;
		int nout_;
};
