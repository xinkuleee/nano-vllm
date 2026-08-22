import os
from glob import glob
import torch
from torch import nn
from safetensors import safe_open


def default_weight_loader(param: nn.Parameter, loaded_weight: torch.Tensor):
    param.data.copy_(loaded_weight)


def load_model(model: nn.Module, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    parameter_names = getattr(model, "checkpoint_parameter_names", lambda name: (name,))
    loaded_parameters: set[str] = set()
    for file in glob(os.path.join(path, "*.safetensors")):
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                for k in packed_modules_mapping:
                    if k in weight_name:
                        v, shard_id = packed_modules_mapping[k]
                        mapped_name = weight_name.replace(k, v)
                        for param_name in parameter_names(mapped_name):
                            param = model.get_parameter(param_name)
                            weight_loader = getattr(param, "weight_loader")
                            weight_loader(param, loaded_weight, shard_id)
                            loaded_parameters.add(param_name)
                        break
                else:
                    for param_name in parameter_names(weight_name):
                        param = model.get_parameter(param_name)
                        weight_loader = getattr(param, "weight_loader", default_weight_loader)
                        weight_loader(param, loaded_weight)
                        loaded_parameters.add(param_name)
    # Models that introduce parameters absent from their source checkpoint can
    # opt into strict validation without changing the established dense-model
    # loading behaviour.
    validator = getattr(model, "validate_loaded_parameters", None)
    if validator is not None:
        validator(frozenset(loaded_parameters))
