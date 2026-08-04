import atexit
from dataclasses import fields
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp

from nanovllm.config import Config
from nanovllm.sampling_params import SamplingParams
from nanovllm.engine.sequence import Sequence
from nanovllm.engine.scheduler import Scheduler
from nanovllm.engine.model_runner import ModelRunner
from nanovllm.engine.contracts import (
    EngineStepResult,
    RequestOutput,
    RunnerBatch,
)


class LLMEngine:

    def __init__(self, model, **kwargs):
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        config = Config(model, **config_kwargs)
        self.config = config
        self._closed = False
        self.ps = []
        self.events = []
        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)
        self.model_runner = ModelRunner(config, 0, self.events)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        self.scheduler = Scheduler(config)
        atexit.register(self.exit)

    def exit(self):
        if self._closed:
            return
        self._closed = True
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.exit()

    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams):
        if isinstance(prompt, str):
            prompt = self.tokenizer.encode(prompt)
        seq = Sequence(prompt, sampling_params, self.config.kvcache_block_size)
        self.scheduler.add(seq)
        return seq.seq_id

    def step(self) -> EngineStepResult:
        scheduled = self.scheduler.schedule()
        runner_output = self.model_runner.call(
            "run",
            RunnerBatch.from_schedule(scheduled),
        )
        if runner_output is None:
            raise RuntimeError("rank-zero runner did not return a result")
        self.scheduler.postprocess(scheduled, runner_output)
        outputs = tuple(
            RequestOutput(seq.seq_id, tuple(seq.completion_token_ids))
            for seq in scheduled.sequences
            if seq.is_finished
        )
        return EngineStepResult(
            outputs=outputs,
            mode=scheduled.mode,
            num_scheduled_tokens=scheduled.num_scheduled_tokens,
            preempted_seq_ids=scheduled.preempted_seq_ids,
            cache_stats=self.scheduler.cache_manager.stats,
        )

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        use_tqdm: bool = True,
    ) -> list[dict[str, str | list[int]]]:
        pbar = tqdm(total=len(prompts), desc="Generating", dynamic_ncols=True, disable=not use_tqdm)
        if not isinstance(sampling_params, list):
            sampling_params = [sampling_params] * len(prompts)
        for prompt, sp in zip(prompts, sampling_params):
            self.add_request(prompt, sp)
        outputs = {}
        prefill_throughput = decode_throughput = 0.
        while not self.is_finished():
            t = perf_counter()
            result = self.step()
            if result.mode.is_prefill:
                prefill_throughput = result.num_scheduled_tokens / (perf_counter() - t)
            else:
                decode_throughput = result.num_scheduled_tokens / (perf_counter() - t)
            pbar.set_postfix({
                "Prefill": f"{int(prefill_throughput)}tok/s",
                "Decode": f"{int(decode_throughput)}tok/s",
            })
            for output in result.outputs:
                outputs[output.request_id] = output.token_ids
                pbar.update(1)
        pbar.close()
        outputs = [outputs[seq_id] for seq_id in sorted(outputs.keys())]
        outputs = [
            {
                "text": self.tokenizer.decode(token_ids),
                "token_ids": list(token_ids),
            }
            for token_ids in outputs
        ]
        return outputs
