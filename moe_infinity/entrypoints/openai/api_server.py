# pyright: reportAny=false, reportExplicitAny=false, reportUnknownMemberType=false, reportUnknownVariableType=false, reportUnknownArgumentType=false, reportArgumentType=false, reportUnknownParameterType=false, reportMissingParameterType=false, reportUnusedCallResult=false, reportUnusedVariable=false, reportUntypedFunctionDecorator=false, reportUnannotatedClassAttribute=false, reportUnknownLambdaType=false

import argparse
import asyncio
import importlib
import json
import os
import time
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import torch

try:
    fastapi = importlib.import_module("fastapi")
    uvicorn = importlib.import_module("uvicorn")
    _fastapi_responses = importlib.import_module("fastapi.responses")
    HTTPException = getattr(fastapi, "HTTPException")
    Request = getattr(fastapi, "Request")
    JSONResponse = getattr(_fastapi_responses, "JSONResponse")
    Response = getattr(_fastapi_responses, "Response")
    StreamingResponse = getattr(_fastapi_responses, "StreamingResponse")
except Exception:

    class HTTPException(Exception):
        def __init__(self, status_code: int, detail: str) -> None:
            self.status_code = status_code
            self.detail = detail
            super().__init__(detail)

    class Request:  # type: ignore[no-redef]
        async def is_disconnected(self) -> bool:
            return False

    class Response:  # type: ignore[no-redef]
        def __init__(self, status_code: int = 200, content: Any = None) -> None:
            self.status_code = status_code
            self.content = content

    class JSONResponse(Response):  # type: ignore[no-redef]
        pass

    class StreamingResponse(Response):  # type: ignore[no-redef]
        def __init__(
            self,
            content: Any,
            media_type: Optional[str] = None,
            status_code: int = 200,
        ) -> None:
            super().__init__(status_code=status_code, content=content)
            self.media_type = media_type

    class _FallbackFastAPI:
        def on_event(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

        def get(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

        def post(self, *_args: Any, **_kwargs: Any) -> Any:
            def _decorator(func: Any) -> Any:
                return func

            return _decorator

    fastapi = SimpleNamespace(FastAPI=_FallbackFastAPI)
    uvicorn = SimpleNamespace(run=lambda *args, **kwargs: None)

from moe_infinity import MoE
from moe_infinity.engine.types import SamplingParams

try:
    from moe_infinity.engine.generation_loop import (
        KVCacheAllocationError,
        PromptTooLongError,
    )
except Exception:
    KVCacheAllocationError = RuntimeError
    PromptTooLongError = ValueError

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    UsageInfo,
    random_uuid,
)

TIMEOUT_KEEP_ALIVE = 5

app = fastapi.FastAPI()

model_name: Optional[str] = None
model: Optional[object] = None
tokenizer: Optional[object] = None
runtime_max_seq_length: int = 4096

request_queue: Optional[asyncio.Queue["_QueuedRequest"]] = None
_worker_task: Optional[asyncio.Task[None]] = None
_shutdown_event: Optional[asyncio.Event] = None


@dataclass
class _QueuedRequest:
    request_id: str
    prompt_token_ids: list[int]
    sampling_params: SamplingParams
    future: asyncio.Future[dict[str, Any]]


class _QueueHTTPError(RuntimeError):
    def __init__(self, status_code: int, detail: str) -> None:
        self.status_code = status_code
        self.detail = detail
        super().__init__(detail)


def initialize_runtime(
    *,
    moe_model: object,
    model_name: Optional[str],
    tokenizer: Optional[object],
    max_seq_length: int,
    device_memory_ratio: float,
    kv_cache_ratio: float,
    max_batch_size: int,
    enable_prefix_caching: bool,
    offload_dir: Optional[str],
) -> None:
    _ = (
        device_memory_ratio,
        kv_cache_ratio,
        max_batch_size,
        enable_prefix_caching,
        offload_dir,
    )
    globals()["model"] = moe_model
    globals()["model_name"] = model_name
    globals()["tokenizer"] = tokenizer
    globals()["runtime_max_seq_length"] = int(max_seq_length)


def parse_prompt_format(prompt: Any) -> tuple[bool, list[Any]]:
    prompt_is_tokens = False
    prompts: list[Any] = [prompt]
    if isinstance(prompt, list):
        if len(prompt) == 0:
            raise ValueError("please provide at least one prompt")
        if isinstance(prompt[0], str):
            prompts = prompt
        elif isinstance(prompt[0], int):
            prompt_is_tokens = True
            prompts = [prompt]
        elif (
            isinstance(prompt[0], list)
            and prompt[0]
            and isinstance(prompt[0][0], int)
        ):
            prompt_is_tokens = True
            prompts = prompt
        else:
            raise ValueError(
                "prompt must be a string, array of strings, array of tokens, or array of token arrays"
            )
    return prompt_is_tokens, prompts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE-Infinity OpenAI-Compatible RESTful API server."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload-dir", type=str, required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--kv-cache-ratio", type=float, default=0.25)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    return parser.parse_args()


def _extract_stop_sequences(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def _build_sampling_params(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    stop: Any,
) -> SamplingParams:
    resolved_max_tokens = (
        int(max_tokens)
        if isinstance(max_tokens, int) and max_tokens > 0
        else SamplingParams().max_tokens
    )
    return SamplingParams(
        temperature=float(temperature) if temperature is not None else 1.0,
        top_p=float(top_p) if top_p is not None else 1.0,
        max_tokens=resolved_max_tokens,
        stop_sequences=tuple(_extract_stop_sequences(stop)),
    )


def _tokenize_text(prompt: str) -> list[int]:
    if tokenizer is None:
        raise _QueueHTTPError(503, "tokenizer not initialized")
    encode_fn = getattr(tokenizer, "encode", None)
    if not callable(encode_fn):
        raise _QueueHTTPError(500, "tokenizer.encode unavailable")
    encoded = encode_fn(prompt)
    if not isinstance(encoded, list):
        raise _QueueHTTPError(500, "failed to tokenize prompt")
    return [int(token_id) for token_id in encoded]


def _chat_prompt_to_token_ids(request: ChatCompletionRequest) -> list[int]:
    if isinstance(request.messages, str):
        return _tokenize_text(request.messages)

    if tokenizer is None:
        raise _QueueHTTPError(503, "tokenizer not initialized")

    apply_chat_template_fn = getattr(tokenizer, "apply_chat_template", None)
    if callable(apply_chat_template_fn):
        prompt = apply_chat_template_fn(
            conversation=request.messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        return _tokenize_text(str(prompt))

    rendered = "\n".join(
        f"{message.get('role', 'user')}: {message.get('content', '')}"
        for message in request.messages
    )
    rendered += "\nassistant:"
    return _tokenize_text(rendered)


def _decode_token_text(token_id: int) -> str:
    if tokenizer is None:
        return ""
    decode_fn = getattr(tokenizer, "decode", None)
    if not callable(decode_fn):
        return ""
    try:
        decoded = decode_fn([token_id], skip_special_tokens=False)
    except TypeError:
        decoded = decode_fn([token_id])
    except Exception:
        return ""
    if isinstance(decoded, str):
        return decoded
    return ""


def _validate_prompt_length(prompt_token_ids: list[int]) -> None:
    if len(prompt_token_ids) > runtime_max_seq_length:
        raise _QueueHTTPError(
            400,
            (
                f"prompt length {len(prompt_token_ids)} exceeds "
                f"max_seq_length {runtime_max_seq_length}"
            ),
        )


def _extract_generated_token_ids(
    output: object, prompt_token_ids: list[int]
) -> list[int]:
    output_token_ids = getattr(output, "output_token_ids", None)
    if isinstance(output_token_ids, list):
        return [int(token_id) for token_id in output_token_ids]

    if isinstance(output, torch.Tensor):
        if output.ndim != 2 or output.shape[0] == 0:
            raise RuntimeError("model.generate returned invalid tensor shape")
        sequence = [int(token_id) for token_id in output[0].tolist()]
        prompt_len = len(prompt_token_ids)
        return sequence[prompt_len:]

    raise RuntimeError("unsupported generation output type")


def _map_generation_exception(exc: Exception) -> Exception:
    if isinstance(exc, _QueueHTTPError):
        return exc
    if isinstance(exc, PromptTooLongError):
        return _QueueHTTPError(400, str(exc))
    if isinstance(exc, KVCacheAllocationError):
        return _QueueHTTPError(503, str(exc))

    message = str(exc)
    lowered = message.lower()
    if "exceeds max_seq_length" in message:
        return _QueueHTTPError(400, message)
    if (
        "failed to allocate kv blocks" in lowered
        or "out of memory" in lowered
        or "cuda out of memory" in lowered
    ):
        return _QueueHTTPError(503, message)
    return exc


def _run_generation_sync(item: _QueuedRequest) -> dict[str, Any]:
    if model is None:
        raise _QueueHTTPError(503, "model not initialized")

    _validate_prompt_length(item.prompt_token_ids)

    native_engine = getattr(model, "_native_generation_engine", None)
    use_native = bool(getattr(model, "use_native_engine", False))

    try:
        if use_native and native_engine is not None:
            output = native_engine.generate(
                prompt_token_ids=item.prompt_token_ids,
                sampling_params=item.sampling_params,
                request_id=item.request_id,
            )
        else:
            input_ids = torch.tensor([item.prompt_token_ids], dtype=torch.long)
            output = getattr(model, "generate")(
                input_ids,
                temperature=item.sampling_params.temperature,
                top_p=item.sampling_params.top_p,
                top_k=item.sampling_params.top_k,
                max_new_tokens=item.sampling_params.max_tokens,
            )
    except Exception as exc:
        raise _map_generation_exception(exc) from exc

    generated_token_ids = _extract_generated_token_ids(
        output=output,
        prompt_token_ids=item.prompt_token_ids,
    )
    token_texts = [
        _decode_token_text(token_id) for token_id in generated_token_ids
    ]

    prompt_tokens = len(item.prompt_token_ids)
    completion_tokens = len(generated_token_ids)
    return {
        "output_text": "".join(token_texts),
        "token_texts": token_texts,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": prompt_tokens + completion_tokens,
    }


async def _worker_loop() -> None:
    while _shutdown_event is not None and not _shutdown_event.is_set():
        if request_queue is None:
            await asyncio.sleep(0.01)
            continue

        try:
            item = await asyncio.wait_for(request_queue.get(), timeout=0.1)
        except asyncio.TimeoutError:
            continue

        try:
            result = await asyncio.to_thread(_run_generation_sync, item)
            if not item.future.done():
                item.future.set_result(result)
        except Exception as exc:
            mapped_exc = _map_generation_exception(exc)
            if not item.future.done():
                item.future.set_exception(mapped_exc)
        finally:
            request_queue.task_done()


async def _ensure_runtime_ready() -> asyncio.Queue[_QueuedRequest]:
    global request_queue
    global _worker_task
    global _shutdown_event

    if request_queue is None:
        request_queue = asyncio.Queue()

    if _shutdown_event is None or _shutdown_event.is_set():
        _shutdown_event = asyncio.Event()

    if _worker_task is None or _worker_task.done():
        _worker_task = asyncio.create_task(_worker_loop())

    return request_queue


async def _submit_generation(
    *,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
) -> dict[str, Any]:
    queue = await _ensure_runtime_ready()
    loop = asyncio.get_running_loop()
    future: asyncio.Future[dict[str, Any]] = loop.create_future()
    await queue.put(
        _QueuedRequest(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            future=future,
        )
    )
    return await future


async def _token_event_stream(token_texts: list[str]) -> Any:
    for token_text in token_texts:
        payload = {"choices": [{"delta": {"content": token_text}}]}
        yield f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
        await asyncio.sleep(0)
    yield "data: [DONE]\n\n"


@app.on_event("startup")
async def startup_event() -> None:
    await _ensure_runtime_ready()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _worker_task

    if _shutdown_event is not None:
        _shutdown_event.set()

    if _worker_task is not None:
        await _worker_task
        _worker_task = None


@app.get("/health")
async def health() -> Response:
    return Response(status_code=200)


@app.post("/v1/completions")
async def completion(request: CompletionRequest, raw_request: Request):
    _ = raw_request
    created_time = int(time.monotonic())
    resolved_model_name = request.model or model_name or "unknown"

    prompt_is_tokens, prompts = parse_prompt_format(request.prompt)
    sampling_params = _build_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )

    if request.stream:
        if len(prompts) != 1:
            return JSONResponse(
                status_code=400,
                content={
                    "error": "streaming completion only supports a single prompt"
                },
            )

        try:
            prompt = prompts[0]
            prompt_token_ids = (
                [int(token_id) for token_id in prompt]
                if prompt_is_tokens
                else _tokenize_text(str(prompt))
            )
            _validate_prompt_length(prompt_token_ids)

            request_id = random_uuid()
            result = await _submit_generation(
                request_id=request_id,
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        except _QueueHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

        return StreamingResponse(
            _token_event_stream(result["token_texts"]),
            media_type="text/event-stream",
        )

    choices: list[CompletionResponseChoice] = []
    usage_prompt_tokens = 0
    usage_completion_tokens = 0

    for index, prompt in enumerate(prompts):
        try:
            prompt_token_ids = (
                [int(token_id) for token_id in prompt]
                if prompt_is_tokens
                else _tokenize_text(str(prompt))
            )
            _validate_prompt_length(prompt_token_ids)

            result = await _submit_generation(
                request_id=random_uuid(),
                prompt_token_ids=prompt_token_ids,
                sampling_params=sampling_params,
            )
        except _QueueHTTPError as exc:
            raise HTTPException(status_code=exc.status_code, detail=exc.detail)

        choices.append(
            CompletionResponseChoice(
                index=index,
                text=result["output_text"],
                logprobs=None,
                finish_reason="stop",
            )
        )
        usage_prompt_tokens += int(result["prompt_tokens"])
        usage_completion_tokens += int(result["completion_tokens"])

    return CompletionResponse(
        id=f"cmpl-{random_uuid()}",
        created=created_time,
        model=resolved_model_name,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=usage_prompt_tokens,
            completion_tokens=usage_completion_tokens,
            total_tokens=usage_prompt_tokens + usage_completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, raw_request: Request):
    _ = raw_request
    created_time = int(time.monotonic())
    resolved_model_name = request.model or model_name or "unknown"
    request_id = random_uuid()

    sampling_params = _build_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )
    try:
        prompt_token_ids = _chat_prompt_to_token_ids(request)
        _validate_prompt_length(prompt_token_ids)

        result = await _submit_generation(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
        )
    except _QueueHTTPError as exc:
        raise HTTPException(status_code=exc.status_code, detail=exc.detail)

    if request.stream:
        return StreamingResponse(
            _token_event_stream(result["token_texts"]),
            media_type="text/event-stream",
        )

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=created_time,
        model=resolved_model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(
                    role="assistant", content=result["output_text"]
                ),
                finish_reason="stop",
            )
        ],
        usage=UsageInfo(
            prompt_tokens=int(result["prompt_tokens"]),
            completion_tokens=int(result["completion_tokens"]),
            total_tokens=int(result["total_tokens"]),
        ),
    )


if __name__ == "__main__":
    from transformers import AutoTokenizer

    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True
    )
    model_name = args.model

    config = {
        "offload_path": os.path.join(args.offload_dir, args.model),
        "device_memory_ratio": args.device_memory_ratio,
        "kv_cache_memory_ratio": args.kv_cache_ratio,
        "use_native_engine": True,
    }
    if args.enable_prefix_caching:
        config["enable_prefix_caching"] = True

    model = MoE(args.model, config)
    max_seq_len = int(getattr(model, "max_seq_length", 4096))

    initialize_runtime(
        moe_model=model,
        model_name=model_name,
        tokenizer=tokenizer,
        max_seq_length=max_seq_len,
        device_memory_ratio=args.device_memory_ratio,
        kv_cache_ratio=args.kv_cache_ratio,
        max_batch_size=args.max_batch_size,
        enable_prefix_caching=args.enable_prefix_caching,
        offload_dir=args.offload_dir,
    )

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
