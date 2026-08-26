import pickle
import torch
import torch.distributed as dist
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory

from nanovllm.config import Config
from nanovllm.engine.contracts import (
    ExecutionMode,
    RunnerBatch,
    RunnerOutput,
    RunnerSequenceInput,
    SequenceUpdate,
)
from nanovllm.engine.model_input import ModelInput
from nanovllm.models.registry import create_model
from nanovllm.layers.sampler import Sampler
from nanovllm.utils.context import (
    AttentionMetadata,
    forward_context,
    get_context,
)
from nanovllm.utils.loader import load_model


class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event]):
        self.config = config
        hf_config = config.hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.world_size = config.tensor_parallel_size
        self.rank = rank
        self.event = event

        dist.init_process_group("nccl", "tcp://localhost:2333", world_size=self.world_size, rank=rank)
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        torch.set_default_dtype(hf_config.dtype)
        torch.set_default_device("cuda")
        self.model = create_model(hf_config, config.attention_backend)
        load_model(self.model, config.model)
        self.sampler = Sampler()
        self.warmup_model()
        self.allocate_kv_cache()
        if not self.enforce_eager:
            self.capture_cudagraph()
        torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)

        if self.world_size > 1:
            if rank == 0:
                self.shm = SharedMemory(name="nanovllm", create=True, size=2**20)
                dist.barrier()
            else:
                dist.barrier()
                self.shm = SharedMemory(name="nanovllm")
                self.loop()

    def exit(self):
        if self.world_size > 1:
            self.shm.close()
            dist.barrier()
            if self.rank == 0:
                self.shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            method_name, args = self.read_shm()
            self.call(method_name, *args)
            if method_name == "exit":
                break

    def read_shm(self):
        assert self.world_size > 1 and self.rank > 0
        self.event.wait()
        n = int.from_bytes(self.shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.world_size > 1 and self.rank == 0
        data = pickle.dumps([method_name, *args])
        n = len(data)
        if n + 4 > len(self.shm.buf):
            raise ValueError(
                "tensor-parallel command exceeds shared-memory capacity: "
                f"{n + 4} > {len(self.shm.buf)} bytes"
            )
        self.shm.buf[0:4] = n.to_bytes(4, "little")
        self.shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        if method is None or not callable(method):
            raise ValueError(f"unknown model-runner method: {method_name}")
        if self.world_size > 1 and self.rank == 0:
            self.write_shm(method_name, *args)
        return method(*args)

    def warmup_model(self):
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        seq_len = min(max_num_batched_tokens, max_model_len)
        num_seqs = min(max_num_batched_tokens // seq_len, self.config.max_num_seqs)
        batch = RunnerBatch(
            ExecutionMode.PREFILL,
            tuple(
                RunnerSequenceInput(
                    request_id=-(index + 1),
                    token_ids=(0,) * seq_len,
                    token_start=0,
                    context_len=seq_len,
                    block_table=(),
                    block_size=self.block_size,
                    temperature=1.0,
                )
                for index in range(num_seqs)
            ),
        )
        self.execute_model(batch)
        torch.cuda.empty_cache()

    def allocate_kv_cache(self):
        config = self.config
        hf_config = config.hf_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        num_kv_heads = hf_config.num_key_value_heads // self.world_size
        head_dim = getattr(hf_config, "head_dim", hf_config.hidden_size // hf_config.num_attention_heads)
        block_bytes = 2 * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.dtype.itemsize
        local_num_blocks = (
            int(total * config.gpu_memory_utilization - used - peak + current)
            // block_bytes
        )
        if self.world_size > 1:
            # The scheduler on rank zero emits physical block IDs used by every
            # tensor-parallel rank.  Base its capacity on the least-free GPU so
            # an ID valid on rank zero cannot overrun another rank's cache.
            shared_capacity = torch.tensor(local_num_blocks, dtype=torch.int64)
            dist.all_reduce(shared_capacity, op=dist.ReduceOp.MIN)
            local_num_blocks = int(shared_capacity.item())
        if local_num_blocks <= 0:
            raise RuntimeError(
                "insufficient GPU memory for one KV-cache block; lower "
                "max_model_len/model size or raise gpu_memory_utilization"
            )
        config.num_kvcache_blocks = local_num_blocks
        self.kv_cache = torch.empty(2, hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                if layer_id >= hf_config.num_hidden_layers:
                    raise RuntimeError("model exposes more KV-cache layers than configured")
                module.k_cache = self.kv_cache[0, layer_id]
                module.v_cache = self.kv_cache[1, layer_id]
                layer_id += 1
        if layer_id != hf_config.num_hidden_layers:
            raise RuntimeError(
                "model KV-cache layer count does not match num_hidden_layers: "
                f"{layer_id} != {hf_config.num_hidden_layers}"
            )

    def prepare_block_tables(self, entries: tuple[RunnerSequenceInput, ...]):
        max_len = max(len(entry.block_table) for entry in entries)
        block_tables = [
            list(entry.block_table) + [-1] * (max_len - len(entry.block_table))
            for entry in entries
        ]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def prepare_prefill(self, batch: RunnerBatch):
        if any(entry.block_size != self.block_size for entry in batch.entries):
            raise ValueError("runner entry block size does not match the model runner")
        input_ids = []
        positions = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        for entry in batch.entries:
            start = entry.token_start
            seqlen_q = entry.token_count
            end = entry.token_end
            seqlen_k = end
            input_ids.extend(entry.token_ids)
            positions.extend(range(start, end))
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            if not entry.block_table:    # warmup
                continue
            start_block = start // self.block_size
            end_block = (end + self.block_size - 1) // self.block_size
            for i in range(start_block, end_block):
                slot_start = entry.block_table[i] * self.block_size
                if i == start_block:
                    slot_start += start % self.block_size
                if i != end_block - 1:
                    slot_end = entry.block_table[i] * self.block_size + self.block_size
                else:
                    slot_end = entry.block_table[i] * self.block_size + end - i * self.block_size
                slot_mapping.extend(range(slot_start, slot_end))
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables = self.prepare_block_tables(batch.entries)
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        metadata = AttentionMetadata(
            mode=ExecutionMode.PREFILL,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            slot_mapping=slot_mapping,
            block_tables=block_tables,
        )
        return ModelInput(input_ids, positions, metadata)

    def prepare_decode(self, batch: RunnerBatch):
        if any(entry.block_size != self.block_size for entry in batch.entries):
            raise ValueError("runner entry block size does not match the model runner")
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        for entry in batch.entries:
            input_ids.append(entry.last_token)
            positions.append(entry.context_len - 1)
            context_lens.append(entry.context_len)
            slot_mapping.append(
                entry.block_table[-1] * self.block_size
                + entry.last_block_num_tokens
                - 1
            )
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_tables = self.prepare_block_tables(batch.entries)
        metadata = AttentionMetadata(
            mode=ExecutionMode.DECODE,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
        )
        return ModelInput(input_ids, positions, metadata)

    def prepare_sample(self, entries: tuple[RunnerSequenceInput, ...]):
        temperatures = [entry.temperature for entry in entries]
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor, is_prefill: bool):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            return self.model.compute_logits(self.model(input_ids, positions))
        else:
            bs = input_ids.size(0)
            context = get_context()
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions
            graph_vars["slot_mapping"].fill_(-1)
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"].zero_()
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"].fill_(-1)
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def execute_model(self, batch: RunnerBatch) -> list[int] | None:
        model_input = (
            self.prepare_prefill(batch)
            if batch.mode.is_prefill
            else self.prepare_decode(batch)
        )
        temperatures = self.prepare_sample(batch.entries) if self.rank == 0 else None
        with forward_context(model_input.attention_metadata):
            logits = self.run_model(
                model_input.input_ids,
                model_input.positions,
                batch.mode.is_prefill,
            )
            token_ids = self.sampler(logits, temperatures).tolist() if self.rank == 0 else None
        return token_ids

    def run(self, batch: RunnerBatch) -> RunnerOutput | None:
        token_ids = self.execute_model(batch)
        if self.rank != 0:
            return None
        if token_ids is None:
            raise RuntimeError("rank-zero sampler did not return token IDs")
        if len(token_ids) != len(batch.entries):
            raise RuntimeError("sampler output count does not match runner batch")
        return RunnerOutput(tuple(
            SequenceUpdate(
                request_id=entry.request_id,
                token_id=token_id,
            )
            for entry, token_id in zip(batch.entries, token_ids)
        ))

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = config.hf_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)
        self.graph_bs = [size for size in (1, 2, 4, 8) if size <= max_bs]
        self.graph_bs.extend(range(16, max_bs + 1, 16))
        # ``max_num_seqs`` is user-configurable and need not be a power of two
        # or a multiple of 16.  Capture its exact upper bound so lookup cannot
        # fail for, for example, a 9- or 17-sequence decode batch.
        if self.graph_bs[-1] != max_bs:
            self.graph_bs.append(max_bs)
        self.graphs = {}
        self.graph_pool = None

        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            metadata = AttentionMetadata(
                mode=ExecutionMode.DECODE,
                slot_mapping=slot_mapping[:bs],
                context_lens=context_lens[:bs],
                block_tables=block_tables[:bs],
            )
            with forward_context(metadata):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # warmup
                with torch.cuda.graph(graph, self.graph_pool):
                    outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
