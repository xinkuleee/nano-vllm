from dataclasses import dataclass

import torch

from nanovllm.utils.context import AttentionMetadata


@dataclass(frozen=True, slots=True)
class ModelInput:
    """Device-resident tensors and metadata for one model invocation."""

    input_ids: torch.Tensor
    positions: torch.Tensor
    attention_metadata: AttentionMetadata
