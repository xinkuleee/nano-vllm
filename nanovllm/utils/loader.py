import os
from glob import glob
from typing import Any


def default_weight_loader(param: Any, loaded_weight: Any):
    param.data.copy_(loaded_weight)


def _replace_path_component(
    parameter_name: str, source_component: str, target_component: str
) -> str | None:
    """Replace one complete dotted path component, never a substring."""

    components = parameter_name.split(".")
    try:
        index = components.index(source_component)
    except ValueError:
        return None
    components[index] = target_component
    return ".".join(components)


def _map_packed_parameter(
    parameter_name: str, packed_modules_mapping: dict[str, tuple[str, Any]]
) -> tuple[str, Any] | None:
    """Map a source checkpoint shard to one runtime packed parameter."""

    for source_component, (target_component, shard_id) in (
        packed_modules_mapping.items()
    ):
        mapped_name = _replace_path_component(
            parameter_name, source_component, target_component
        )
        if mapped_name is not None:
            return mapped_name, shard_id
    return None


def _load_complete_or_sharded_weight(param: Any, loaded_weight: Any) -> None:
    """Copy a runtime-format tensor or shard a full source tensor."""

    if param.shape == loaded_weight.shape:
        default_weight_loader(param, loaded_weight)
        return
    weight_loader = getattr(param, "weight_loader", default_weight_loader)
    weight_loader(param, loaded_weight)


def load_model(model: Any, path: str):
    packed_modules_mapping = getattr(model, "packed_modules_mapping", {})
    parameter_names = getattr(model, "checkpoint_parameter_names", lambda name: (name,))
    loaded_parameters: set[str] = set()
    files = sorted(glob(os.path.join(path, "*.safetensors")))
    if not files:
        raise FileNotFoundError(f"no safetensors checkpoint files found in: {path}")

    from safetensors import safe_open

    for file in files:
        with safe_open(file, "pt", "cpu") as f:
            for weight_name in f.keys():
                loaded_weight = f.get_tensor(weight_name)
                packed = _map_packed_parameter(weight_name, packed_modules_mapping)
                if packed is None:
                    for param_name in parameter_names(weight_name):
                        param = model.get_parameter(param_name)
                        # Native runtime checkpoints already store complete
                        # packed QKV/gate-up tensors.  Copying directly avoids
                        # calling their source-shard loader without a shard ID.
                        # A full Hugging Face tensor loaded under tensor
                        # parallelism has a different shape and still needs its
                        # normal per-parameter sharding loader.
                        _load_complete_or_sharded_weight(param, loaded_weight)
                        loaded_parameters.add(param_name)
                    continue
                mapped_name, shard_id = packed
                for param_name in parameter_names(mapped_name):
                    param = model.get_parameter(param_name)
                    weight_loader = getattr(param, "weight_loader")
                    weight_loader(param, loaded_weight, shard_id)
                    loaded_parameters.add(param_name)
    # Models that introduce parameters absent from their source checkpoint can
    # opt into strict validation without changing the established dense-model
    # loading behaviour.
    validator = getattr(model, "validate_loaded_parameters", None)
    if validator is not None:
        validator(frozenset(loaded_parameters))
