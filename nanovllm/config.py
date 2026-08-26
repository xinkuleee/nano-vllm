import os
from dataclasses import dataclass
from typing import Any


@dataclass(slots=True)
class Config:
    model: str
    max_num_batched_tokens: int = 16384
    max_num_seqs: int = 512
    max_model_len: int = 4096
    gpu_memory_utilization: float = 0.9
    tensor_parallel_size: int = 1
    enforce_eager: bool = False
    hf_config: Any | None = None
    eos: int = -1
    kvcache_block_size: int = 256
    num_kvcache_blocks: int = -1
    attention_backend: str = "flash_attention"

    def __post_init__(self):
        integer_fields = {
            "max_num_batched_tokens": self.max_num_batched_tokens,
            "max_num_seqs": self.max_num_seqs,
            "max_model_len": self.max_model_len,
            "kvcache_block_size": self.kvcache_block_size,
            "tensor_parallel_size": self.tensor_parallel_size,
        }
        for name, value in integer_fields.items():
            if not isinstance(value, int) or isinstance(value, bool):
                raise TypeError(f"{name} must be an integer")
        if not os.path.isdir(self.model):
            raise ValueError(f"model directory does not exist: {self.model}")
        if self.max_num_batched_tokens <= 0:
            raise ValueError("max_num_batched_tokens must be positive")
        if self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be positive")
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be positive")
        if not 0 < self.gpu_memory_utilization < 1:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.kvcache_block_size <= 0 or self.kvcache_block_size % 256 != 0:
            raise ValueError("kvcache_block_size must be a positive multiple of 256")
        if not 1 <= self.tensor_parallel_size <= 8:
            raise ValueError("tensor_parallel_size must be between 1 and 8")

        from transformers import AutoConfig

        self.hf_config = AutoConfig.from_pretrained(self.model)
        self.max_model_len = min(self.max_model_len, self.hf_config.max_position_embeddings)
        implementation = getattr(self.hf_config, "mini_moe_implementation", None)
        if (
            implementation in {"sparse_dispatch", "triton_grouped"}
            and self.tensor_parallel_size != 1
        ):
            raise ValueError(
                "Triton Mini-MoE grouped dispatch currently requires "
                "tensor_parallel_size=1"
            )
        if (
            implementation in {"sparse_reference", "sparse_dispatch", "triton_grouped"}
            and not self.enforce_eager
        ):
            raise ValueError(
                "sparse Mini-MoE dispatch uses dynamic token shapes and requires "
                "enforce_eager=True; use dense_masked for CUDA Graph capture"
            )
