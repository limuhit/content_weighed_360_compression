#pragma once
#include "ext_all.hpp" 
#include "timer.h"
#include "base_opt.hpp"
#include <map>
#include <sstream> //for std::stringstream 
#include <string>  //for std::string
class pseudo_context_opt: public base_opt{
	public:
		pseudo_context_opt(int npart, int rt, int device = 0, bool timeit=false){
			npart_ = npart;
			rt_ = rt;
			base_opt_init(device,timeit);
		}
		~pseudo_context_opt(){}
		void init();
		bool reshape_hw(int num, int height, int width);
        bool reshape_channel_pad(int channel, int pad);
        void start_context(int num, int width, at::Tensor weight)
        {

            if(width!=data_width_ || num!=data_num_){
                param_.clear();
                param2_.clear();
                hindex_.clear();
                hindex2_.clear();
                stride_inv_.clear();
                width_dict_.clear();
                pad_channel_dict_.clear();
            }
            
            pad_channel_dict_update_.clear();
            width_dict_update_.clear();
            data_width_ = width;
            data_num_ = num;
            weight_ = weight;
            //printf("here,clear\n");
        }
		std::vector<at::Tensor> produce_param(int num, int channel, int height, int width, int pad);
        at::Tensor produce_param_fill(int num, int height, int width);
        std::map<int, at::Tensor> param_, param2_, hindex_, hindex2_, index_,  stride_inv_;
        std::map<int,int> pad_channel_dict_, width_dict_;
        std::map<int,int> pad_channel_dict_update_, width_dict_update_;
		int npart_, pad_, rt_;
        int data_width_ = -1;
        int data_num_ = -1;
        int cp_;
        at::Tensor weight_;
};


class pseudo_context_shell{
    public:
        pseudo_context_shell(int npart, int rt, int device = 0, bool timeit=false){
            ctx = new pseudo_context_opt(npart,rt,device,timeit);
            const void * address = static_cast<const void*>(ctx);
            std::stringstream ss;
            ss << address;  
            addr = ss.str(); 
        }
        ~pseudo_context_shell(){delete ctx;}
        void to(int device){
            ctx->to(device);
        }
        void start_context(int num, int width, at::Tensor weight){
            ctx->start_context(num, width, weight);
        }
        at::Tensor produce_fill_param(int num, int height,int width){
           return ctx->produce_param_fill(num,height,width);
        }
        std::vector<at::Tensor> produce_pad_param(int num, int channel, int height, int width, int pad){
           return ctx->produce_param(num,channel,height,width,pad);
        }
        std::string get_pointer(){
            return addr;
        }
        pseudo_context_opt * ctx;
        std::string addr;
};