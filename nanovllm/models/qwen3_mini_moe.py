"""Qwen3 variant whose selected dense MLP layers become Mini-MoE layers."""

from __future__ import annotations

import torch
from torch import nn

from nanovllm.layers.backends import AttentionBackend
from nanovllm.layers.mini_moe import MiniMoE
from nanovllm.models.qwen3 import Qwen3ForCausalLM, Qwen3MLP


def mini_moe_layer_ids(config: object) -> tuple[int, ...]:
    raw = getattr(config, "mini_moe_layer_ids", None)
    if raw is None:
        raw = [0]
    layer_ids = tuple(int(layer_id) for layer_id in raw)
    if not layer_ids:
        raise ValueError("mini_moe_layer_ids must contain at least one layer")
    if len(set(layer_ids)) != len(layer_ids):
        raise ValueError("mini_moe_layer_ids contains duplicates")
    if any(layer_id < 0 or layer_id >= config.num_hidden_layers for layer_id in layer_ids):
        raise ValueError("mini_moe_layer_ids contains an out-of-range layer")
    return layer_ids


def _make_expert(config: object) -> Qwen3MLP:
    return Qwen3MLP(
        hidden_size=config.hidden_size,
        intermediate_size=config.intermediate_size,
        hidden_act=config.hidden_act,
    )


class Qwen3MiniMoEForCausalLM(Qwen3ForCausalLM):
    """Qwen3 checkpoint-compatible model with converted MLP layers.

    Dense Qwen MLP tensors are loaded directly into every expert through a
    checkpoint-name expansion hook.  With a zero router and identical experts,
    the weighted expert sum exactly equals the original dense MLP output.
    Only selected layers are duplicated so the experiment fits a 12-GB 3060.
    """

    def __init__(
        self,
        config: object,
        attention_backend: AttentionBackend | str = "flash_attention",
    ) -> None:
        super().__init__(config, attention_backend)
        num_experts = int(getattr(config, "mini_moe_num_experts", 4))
        top_k = int(getattr(config, "mini_moe_top_k", min(2, num_experts)))
        implementation = str(
            getattr(config, "mini_moe_implementation", "dense_masked")
        )
        self.mini_moe_layer_ids = mini_moe_layer_ids(config)
        self._tied_word_embeddings = bool(config.tie_word_embeddings)
        for layer_id in self.mini_moe_layer_ids:
            experts = [_make_expert(config) for _ in range(num_experts)]
            self.model.layers[layer_id].mlp = MiniMoE(
                config.hidden_size,
                experts,
                top_k=top_k,
                implementation=implementation,
            )
            nn.init.zeros_(self.model.layers[layer_id].mlp.router.weight)

    def checkpoint_parameter_names(self, weight_name: str) -> tuple[str, ...]:
        for layer_id in self.mini_moe_layer_ids:
            prefix = f"model.layers.{layer_id}.mlp."
            if weight_name.startswith(prefix):
                suffix = weight_name[len(prefix):]
                return tuple(
                    f"{prefix}experts.{expert_id}.{suffix}"
                    for expert_id in range(self.model.layers[layer_id].mlp.num_experts)
                )
        return (weight_name,)

    def validate_loaded_parameters(self, loaded: frozenset[str]) -> None:
        """Ensure every cloned expert was populated from the dense checkpoint."""

        allowed_missing = {
            f"model.layers.{layer_id}.mlp.router.weight"
            for layer_id in self.mini_moe_layer_ids
        }
        if self._tied_word_embeddings and "model.embed_tokens.weight" in loaded:
            allowed_missing.add("lm_head.weight")
        expected = {name for name, _ in self.named_parameters()}
        missing = expected.difference(loaded, allowed_missing)
        if missing:
            preview = ", ".join(sorted(missing)[:8])
            raise RuntimeError(
                "checkpoint did not initialise Mini-MoE parameters: " + preview
            )

    @torch.no_grad()
    def diversify_experts_(self, noise_std: float, seed: int = 0) -> None:
        """Perturb cloned experts for an inference-only routing experiment.

        This intentionally changes model behaviour and is not training.  Keep
        ``noise_std=0`` when checking parity with the dense checkpoint.
        """

        if noise_std < 0:
            raise ValueError("noise_std cannot be negative")
        generator = torch.Generator(device=next(self.parameters()).device)
        generator.manual_seed(seed)
        for layer_id in self.mini_moe_layer_ids:
            moe = self.model.layers[layer_id].mlp
            for expert_id, expert in enumerate(moe.experts):
                if expert_id == 0 or noise_std == 0:
                    continue
                for parameter in expert.parameters():
                    noise = torch.randn(
                        parameter.shape,
                        dtype=parameter.dtype,
                        device=parameter.device,
                        generator=generator,
                    )
                    parameter.add_(noise * noise_std)
