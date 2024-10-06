#include "main.hpp"
#include <torch/extension.h>
namespace py = pybind11;
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    
    py::class_<projects_opt>(m,"ProjectsOp")
        .def(py::init<int, int , std::vector<float>, std::vector<float>, float, bool, int, bool>())
        .def("to", &projects_opt::to)
        .def("forward", &projects_opt::forward_cuda)
        .def("backward", &projects_opt::backward_cuda);

    py::class_<sphere_slice_opt>(m,"SphereSliceOp")
        .def(py::init<int, int, int, int, bool>())
        .def("to", &sphere_slice_opt::to)
        .def("forward", &sphere_slice_opt::forward_cuda)
        .def("backward", &sphere_slice_opt::backward_cuda);

    py::class_<sphere_uslice_opt>(m,"SphereUsliceOp")
        .def(py::init<int, int, int, int, bool>())
        .def("to", &sphere_uslice_opt::to)
        .def("forward", &sphere_uslice_opt::forward_cuda)
        .def("backward", &sphere_uslice_opt::backward_cuda);

    py::class_<pseudo_context_shell>(m,"PseudoContextOp")
        .def(py::init<int, int, int, bool>())
        .def("to", &pseudo_context_shell::to)
        .def("start_context", &pseudo_context_shell::start_context)
        .def("addr",&pseudo_context_shell::get_pointer)
        .def("produce_fill_param",&pseudo_context_shell::produce_fill_param)
        .def("produce_pad_param",&pseudo_context_shell::produce_pad_param);

    py::class_<pseudo_pad_opt>(m,"PseudoPadOp")
        .def(py::init<int, int, std::string, int, bool>())
        .def("to", &pseudo_pad_opt::to)
        .def("forward", &pseudo_pad_opt::forward_cuda)
        .def("backward", &pseudo_pad_opt::backward_cuda);

    py::class_<pseudo_fill_opt>(m,"PseudoFillOp")
        .def(py::init<int, int, int, int, std::string, int, int, bool>())
        .def("to", &pseudo_fill_opt::to)
        .def("forward", &pseudo_fill_opt::forward_cuda)
        .def("backward", &pseudo_fill_opt::backward_cuda);

    py::class_<pseudo_entropy_context_shell>(m,"PseudoEntropyContextOp")
        .def(py::init<int, int, int, int, bool>())
        .def("to", &pseudo_entropy_context_shell::to)
        .def("start_context", &pseudo_entropy_context_shell::start_context)
        .def("addr",&pseudo_entropy_context_shell::get_pointer);

    py::class_<pseudo_entropy_pad_opt>(m,"PseudoEntropyPadOp")
        .def(py::init<int, int, std::string, int, bool>())
        .def("to", &pseudo_entropy_pad_opt::to)
        .def("forward", &pseudo_entropy_pad_opt::forward_cuda)
        .def("backward", &pseudo_entropy_pad_opt::backward_cuda);
    
    py::class_<pseudo_merge_opt>(m,"PseudoMergeOp")
        .def(py::init<int, std::string, int, int, bool>())
        .def("to", &pseudo_merge_opt::to)
        .def("forward", &pseudo_merge_opt::forward_cuda)
        .def("backward", &pseudo_merge_opt::backward_cuda);

    py::class_<pseudo_split_opt>(m,"PseudoSplitOp")
        .def(py::init<int, std::string, int, int, bool>())
        .def("to", &pseudo_split_opt::to)
        .def("forward", &pseudo_split_opt::forward_cuda)
        .def("backward", &pseudo_split_opt::backward_cuda);
    
    py::class_<entropy_context_shell>(m,"EntropyContextOp")
        .def(py::init<int, int, std::vector<float>, int, bool>())
        .def("to", &entropy_context_shell::to)
        .def("start_context", &entropy_context_shell::start_context)
        .def("addr",&entropy_context_shell::get_pointer);

    py::class_<acc_grad_opt>(m,"AccGradOp")
        .def(py::init<int, bool>())
        .def("to", &acc_grad_opt::to)
        .def("apply", &acc_grad_opt::forward_cuda);
};