from __future__ import annotations

import threading

import torch

from .batch import BatchMetadata
from .model_runner import ModelRunner


class _CapturedGraphState:
    graph: torch.cuda.CUDAGraph
    input_ids: torch.Tensor
    position_ids: torch.Tensor
    attention_mask: torch.Tensor
    logits: torch.Tensor

    def __init__(
        self,
        graph: torch.cuda.CUDAGraph,
        input_ids: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        logits: torch.Tensor,
    ) -> None:
        self.graph = graph
        self.input_ids = input_ids
        self.position_ids = position_ids
        self.attention_mask = attention_mask
        self.logits = logits


class CudaGraphRunner:
    model_runner: ModelRunner
    max_batch_sizes: tuple[int, ...]

    def __init__(
        self,
        model_runner: ModelRunner,
        max_batch_sizes: tuple[int, ...] = (1, 2, 4, 8, 16, 32),
    ) -> None:
        self.model_runner = model_runner
        self.max_batch_sizes = tuple(
            sorted(
                {batch_size for batch_size in max_batch_sizes if batch_size > 0}
            )
        )
        self._graphs: dict[int, _CapturedGraphState] = {}
        self._warmed_batch_sizes: set[int] = set()
        self._lock = threading.Lock()
        self._disabled = (
            __import__("os").getenv("MOE_DISABLE_CUDA_GRAPHS") == "1"
        )

    def is_compatible(self, batch: BatchMetadata) -> bool:
        """Check if batch can use a captured graph (decode-only, matching size)."""
        if not self._is_enabled():
            return False

        batch_size = len(batch.seq_ids)
        return (
            batch_size > 0
            and self._is_decode_batch(batch)
            and batch_size in self._graphs
        )

    def capture(self, batch: BatchMetadata) -> None:
        """Capture CUDA graph for the given batch shape. Requires warmup first."""
        if not self._is_enabled():
            return

        batch_size = self._validate_batch_for_graph(batch)
        with self._lock:
            if batch_size not in self._warmed_batch_sizes:
                raise RuntimeError(
                    "warmup() must be called before capture() for this batch size"
                )

            model_inputs = self.model_runner.prepare_inputs(batch)
            example_logits = self._forward_decode(model_inputs)

            static_input_ids = model_inputs["input_ids"].clone()
            static_position_ids = model_inputs["position_ids"].clone()
            static_attention_mask = model_inputs["attention_mask"].clone()
            static_logits = torch.empty(
                example_logits.shape,
                dtype=example_logits.dtype,
                device=example_logits.device,
            )

            graph = torch.cuda.CUDAGraph()
            torch.cuda.synchronize(self.model_runner.device)
            with torch.cuda.graph(graph):
                captured_logits = self._forward_decode(
                    {
                        "input_ids": static_input_ids,
                        "position_ids": static_position_ids,
                        "attention_mask": static_attention_mask,
                    }
                )
                static_logits.copy_(captured_logits)

            self._graphs[batch_size] = _CapturedGraphState(
                graph=graph,
                input_ids=static_input_ids,
                position_ids=static_position_ids,
                attention_mask=static_attention_mask,
                logits=static_logits,
            )

    def replay(self, batch: BatchMetadata) -> torch.Tensor:
        """Copy inputs into captured buffers and replay graph."""
        if len(batch.seq_ids) == 0 or batch.total_tokens == 0:
            return self.model_runner._empty_logits()

        if not self._is_enabled():
            raise RuntimeError("CUDA graphs are disabled")

        batch_size = self._validate_batch_for_graph(batch)
        with self._lock:
            state = self._graphs.get(batch_size)
            if state is None:
                raise RuntimeError(
                    f"no CUDA graph has been captured for batch size {batch_size}"
                )

            model_inputs = self.model_runner.prepare_inputs(batch)
            state.input_ids.copy_(model_inputs["input_ids"])
            state.position_ids.copy_(model_inputs["position_ids"])
            state.attention_mask.copy_(model_inputs["attention_mask"])
            state.graph.replay()
            return state.logits.clone()

    def warmup(self, batch: BatchMetadata, num_iters: int = 3) -> None:
        """Run warmup iterations before capture."""
        if not self._is_enabled():
            return

        if num_iters < 1:
            raise ValueError("num_iters must be at least 1")

        batch_size = self._validate_batch_for_graph(batch)
        with self._lock:
            for _ in range(num_iters):
                model_inputs = self.model_runner.prepare_inputs(batch)
                _ = self._forward_decode(model_inputs)

            torch.cuda.synchronize(self.model_runner.device)
            self._warmed_batch_sizes.add(batch_size)

    def invalidate(self) -> None:
        """Clear all captured graphs (call on model reload)."""
        with self._lock:
            self._graphs.clear()
            self._warmed_batch_sizes.clear()

    def _forward_decode(
        self, model_inputs: dict[str, torch.Tensor]
    ) -> torch.Tensor:
        eval_fn = getattr(self.model_runner.model, "eval", None)
        if callable(eval_fn):
            _ = eval_fn()

        forward_fn = getattr(self.model_runner.model, "forward", None)
        if not callable(forward_fn):
            raise ValueError("model must define callable forward()")

        with torch.no_grad():
            outputs = forward_fn(
                input_ids=model_inputs["input_ids"],
                position_ids=model_inputs["position_ids"],
                attention_mask=model_inputs["attention_mask"],
                use_cache=True,
            )

        logits = self.model_runner._extract_logits(outputs)
        batch_size = model_inputs["input_ids"].shape[0]
        if logits.dim() == 3:
            if logits.shape[0] != batch_size or logits.shape[1] != 1:
                raise ValueError(
                    "CUDA graph decode capture requires rank-3 logits with seq_len=1"
                )
            return logits[:, 0, :]

        if logits.dim() != 2:
            raise ValueError(
                f"model output logits must be rank-2 or rank-3, got rank-{logits.dim()}"
            )
        if logits.shape[0] != batch_size:
            raise ValueError(
                "decode logits row count must match the decode batch size"
            )
        return logits

    def _validate_batch_for_graph(self, batch: BatchMetadata) -> int:
        if not self._is_cuda_device():
            raise RuntimeError(
                "CUDA graph capture requires a CUDA ModelRunner device"
            )

        batch_size = len(batch.seq_ids)
        if batch_size == 0 or batch.total_tokens == 0:
            raise ValueError(
                "empty batches are not eligible for CUDA graph capture"
            )
        if batch_size not in self.max_batch_sizes:
            raise ValueError(
                f"batch size {batch_size} is not configured for CUDA graph capture"
            )
        if not self._is_decode_batch(batch):
            raise ValueError(
                "CUDA graphs only support decode batches with seq_len=1"
            )
        return batch_size

    def _is_enabled(self) -> bool:
        return not self._disabled and self._is_cuda_device()

    def _is_cuda_device(self) -> bool:
        return (
            self.model_runner.device.type == "cuda"
            and torch.cuda.is_available()
        )

    @staticmethod
    def _is_decode_batch(batch: BatchMetadata) -> bool:
        if len(batch.seq_ids) == 0 or batch.total_tokens == 0:
            return False
        if any(batch.is_prefill):
            return False
        if any(seq_len != 1 for seq_len in batch.seq_lengths):
            return False
        return batch.total_tokens == len(batch.seq_ids)


__all__ = ["CudaGraphRunner"]
