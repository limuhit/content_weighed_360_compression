from typing import Any, Dict, Mapping, cast

import torch.nn as nn
import torch.optim as optim

from compressai.registry import OPTIMIZERS


def net_aux_optimizer(
    net: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, optim.Optimizer]:
    """Returns separate optimizers for net and auxiliary losses.

    Each optimizer operates on a mutually exclusive set of parameters.
    """
    parameters = {
        "net": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles") and name.find("param_net") < 0
        },
        "aux": {
            name
            for name, param in net.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        },
        "param":{
            name
            for name, param in net.named_parameters()
            if param.requires_grad and name.find("param_net") >= 0
        }
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net.named_parameters())
    inter_params = (parameters["net"] & parameters["aux"]) | (parameters["param"] & parameters["aux"]) | (parameters["net"] & parameters["param"]) 
    union_params = parameters["net"] | parameters["aux"] | parameters["param"]
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0

    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net", "aux","param"]}

    return cast(Dict[str, optim.Optimizer], optimizer)

def net_aux_optimizer2(
    net_cmp: nn.Module, net_param: nn.Module, conf: Mapping[str, Any]
) -> Dict[str, optim.Optimizer]:
    """Returns separate optimizers for net and auxiliary losses.

    Each optimizer operates on a mutually exclusive set of parameters.
    """
    parameters = {
        "net": {
            name
            for name, param in net_cmp.named_parameters()
            if param.requires_grad and not name.endswith(".quantiles")
        },
        "aux": {
            name
            for name, param in net_cmp.named_parameters()
            if param.requires_grad and name.endswith(".quantiles")
        },
        "param":{
            name
            for name, param in net_param.named_parameters()
            if param.requires_grad 
        }
    }

    # Make sure we don't have an intersection of parameters
    params_dict = dict(net_cmp.named_parameters())
    inter_params = parameters["net"] & parameters["aux"]
    union_params = parameters["net"] | parameters["aux"] 
    assert len(inter_params) == 0
    assert len(union_params) - len(params_dict.keys()) == 0
    params_dict.update(dict(net_param.named_parameters()))

    def make_optimizer(key):
        kwargs = dict(conf[key])
        del kwargs["type"]
        params = (params_dict[name] for name in sorted(parameters[key]))
        return OPTIMIZERS[conf[key]["type"]](params, **kwargs)

    optimizer = {key: make_optimizer(key) for key in ["net", "aux","param"]}

    return cast(Dict[str, optim.Optimizer], optimizer)