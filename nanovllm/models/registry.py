from collections.abc import Callable
from dataclasses import dataclass

from torch import nn


@dataclass(frozen=True, slots=True)
class ModelBuildContext:
    hf_config: object
    attention_backend: str


ModelFactory = Callable[[ModelBuildContext], nn.Module]


def _create_qwen3(context: ModelBuildContext) -> nn.Module:
    from nanovllm.layers.backends import create_attention_backend
    from nanovllm.models.qwen3 import Qwen3ForCausalLM

    return Qwen3ForCausalLM(
        context.hf_config,
        attention_backend=create_attention_backend(context.attention_backend),
    )


_MODEL_REGISTRY: dict[str, ModelFactory] = {
    "Qwen3ForCausalLM": _create_qwen3,
    "qwen3": _create_qwen3,
}


def register_model(identifier: str, factory: ModelFactory) -> None:
    """Register an architecture without changing the runtime.

    This is the extension point for future dense and MoE model families.
    """

    if identifier in _MODEL_REGISTRY:
        raise ValueError(f"model identifier is already registered: {identifier}")
    _MODEL_REGISTRY[identifier] = factory


def create_model(
    hf_config: object,
    attention_backend: str = "flash_attention",
) -> nn.Module:
    architectures = getattr(hf_config, "architectures", None) or []
    if isinstance(architectures, str):
        architectures = [architectures]
    model_type = getattr(hf_config, "model_type", None)
    candidates = [
        identifier
        for identifier in [*architectures, model_type]
        if isinstance(identifier, str)
    ]
    for identifier in candidates:
        factory = _MODEL_REGISTRY.get(identifier)
        if factory is not None:
            return factory(ModelBuildContext(hf_config, attention_backend))
    supported = ", ".join(sorted(_MODEL_REGISTRY))
    requested = ", ".join(candidates) or "<missing>"
    raise ValueError(
        f"unsupported model architecture: {requested}; supported architectures: {supported}"
    )
