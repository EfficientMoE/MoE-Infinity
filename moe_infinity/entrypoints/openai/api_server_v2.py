import argparse
import asyncio
import importlib
import json
import logging
import os
import time
from contextlib import suppress
from threading import Event, Lock
from types import SimpleNamespace
from typing import Any, Literal, Optional, cast

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

import moe_infinity.serving.watchdog as watchdog_module
from moe_infinity.serving.engine import ContinuousBatchingEngine, RequestOutput
from moe_infinity.serving.health import ServerHealthState
from moe_infinity.serving.sequence import SamplingParams
from moe_infinity.serving.stream import StreamManager
from moe_infinity.serving.validation import (
    ContextLengthExceededError,
    InvalidRequestError,
    validate_context_length,
    validate_required_params,
    validate_sampling_params,
)

from .protocol import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatCompletionResponseChoice,
    ChatMessage,
    CompletionRequest,
    CompletionResponse,
    CompletionResponseChoice,
    CompletionResponseStreamChoice,
    CompletionStreamResponse,
    UsageInfo,
    create_error_response,
    random_uuid,
)

TIMEOUT_KEEP_ALIVE = 5

app = fastapi.FastAPI()
engine: Optional[ContinuousBatchingEngine] = None
stream_manager: Optional[StreamManager] = None
tokenizer: Optional[object] = None
model_name_global: Optional[str] = None
runtime_max_seq_length: int = 4096

_engine_task: Optional[asyncio.Task[None]] = None
_engine_shutdown_event: Optional[asyncio.Event] = None
_model_init_task: Optional[asyncio.Task[None]] = None
_startup_args: Optional[argparse.Namespace] = None
_health_state = ServerHealthState()
_startup_watchdog: Optional[Any] = None
_decode_watchdog: Optional[Any] = None
_watchdog_config: Optional[Any] = None

_contextpilot_enabled: bool = False
_contextpilot_debug: bool = False
_contextpilot_fault: str = "none"
_contextpilot_state_lock = Lock()
_contextpilot_fallback_count: int = 0
_contextpilot_last_fallback_count: int = 0

_cp_logger = logging.getLogger("moe_infinity.contextpilot")
_cp_middleware: Optional[Any] = None


def _resolve_contextpilot_enabled(cli_enabled: bool) -> bool:
    if os.environ.get("CONTEXTPILOT_ENABLED", "1") == "0":
        return False
    return bool(cli_enabled)


def _is_contextpilot_active() -> bool:
    with _contextpilot_state_lock:
        enabled = _contextpilot_enabled
    return enabled and os.environ.get("CONTEXTPILOT_ENABLED", "1") != "0"


def _current_contextpilot_fault() -> str:
    with _contextpilot_state_lock:
        return _contextpilot_fault


def _record_contextpilot_fallback(request_id: str, exc: Exception) -> None:
    with _contextpilot_state_lock:
        global _contextpilot_fallback_count
        global _contextpilot_last_fallback_count
        _contextpilot_fallback_count += 1
        _contextpilot_last_fallback_count = _contextpilot_fallback_count
    _cp_logger.warning(
        "request_id=%s CP middleware error, falling back: %s",
        request_id,
        exc,
    )


def _ensure_cp_middleware_initialized() -> Optional[Any]:
    if not _is_contextpilot_active():
        return None

    with _contextpilot_state_lock:
        global _cp_middleware
        if _cp_middleware is not None:
            return _cp_middleware

    try:
        from moe_infinity.serving.contextpilot_middleware import (
            ContextPilotMiddleware,
        )

        middleware = ContextPilotMiddleware(use_gpu=False)
    except Exception as exc:
        _cp_logger.warning(
            "Failed to initialize ContextPilot middleware: %s",
            exc,
        )
        return None

    with _contextpilot_state_lock:
        if _cp_middleware is None:
            _cp_middleware = middleware
            _cp_logger.info("ContextPilot middleware initialized")
        return _cp_middleware


def _process_chat_messages_with_contextpilot(
    messages: Any,
    *,
    request_id: str,
) -> Any:
    if not isinstance(messages, list):
        return messages
    if not _is_contextpilot_active():
        return messages

    middleware = _ensure_cp_middleware_initialized()
    if middleware is None:
        return messages

    try:
        fault = _current_contextpilot_fault()
        if fault != "none":
            raise RuntimeError(f"CP fault injected: {fault}")

        t0 = time.monotonic()
        processed_messages = middleware.process_chat_request(messages)
        cp_latency_ms = (time.monotonic() - t0) * 1000
        _cp_logger.debug(
            "request_id=%s reorder=%.1fms", request_id, cp_latency_ms
        )
        return processed_messages
    except Exception as exc:
        _record_contextpilot_fallback(request_id, exc)
        return messages


def _process_completion_prompt_with_contextpilot(
    prompt: str,
    *,
    request_id: str,
) -> str:
    if not _is_contextpilot_active():
        return prompt

    middleware = _ensure_cp_middleware_initialized()
    if middleware is None:
        return prompt

    try:
        fault = _current_contextpilot_fault()
        if fault != "none":
            raise RuntimeError(f"CP fault injected: {fault}")

        t0 = time.monotonic()
        optimized_prompt = middleware.process_completion_request(prompt)
        cp_latency_ms = (time.monotonic() - t0) * 1000
        _cp_logger.debug(
            "request_id=%s reorder=%.1fms", request_id, cp_latency_ms
        )
        return optimized_prompt
    except Exception as exc:
        _record_contextpilot_fallback(request_id, exc)
        return prompt


def initialize_with_model(
    *,
    moe_model: object,
    model_name: Optional[str],
    tok: Optional[object],
    max_seq_length: int,
    device_memory_ratio: float = 0.75,
    kv_cache_ratio: float = 0.25,
    max_batch_size: int = 32,
    enable_prefix_caching: bool = False,
) -> None:
    """Initialize the v2 server with a pre-loaded MoE model.

    This is the programmatic entry point used by ``MoE.serve()``.  Instead of
    loading a model from scratch (which ``_initialize_model`` does when the
    server is started via CLI), this function wires a *pre-existing* MoE
    instance into the continuous-batching engine.
    """
    global engine, stream_manager, tokenizer, model_name_global
    global runtime_max_seq_length, _health_state

    hf_model = getattr(moe_model, "model", moe_model)
    offload_engine = getattr(moe_model, "engine", None)

    args = argparse.Namespace(
        device_memory_ratio=device_memory_ratio,
        kv_cache_ratio=kv_cache_ratio,
        max_batch_size=max_batch_size,
        enable_prefix_caching=enable_prefix_caching,
    )
    engine_config = _build_engine_config(args=args, model=hf_model)

    tokenizer = tok
    model_name_global = model_name
    configured_max_seq_length = engine_config.get("max_seq_length")
    if isinstance(configured_max_seq_length, int):
        runtime_max_seq_length = configured_max_seq_length
    else:
        runtime_max_seq_length = int(max_seq_length)

    initialized_engine = ContinuousBatchingEngine(
        model=hf_model,
        engine=offload_engine,
        config=engine_config,
        tokenizer=tokenizer,
    )
    if stream_manager is None:
        stream_manager = StreamManager()

    engine = initialized_engine
    _health_state.set_healthy()


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


def _extract_stop_sequences(stop: Any) -> list[str]:
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    if isinstance(stop, list):
        return [s for s in stop if isinstance(s, str)]
    return []


def _decode_token_text(token_id: int, token_text: Optional[str]) -> str:
    if token_text is not None:
        return token_text
    if tokenizer is None:
        return ""
    decode_fn = getattr(tokenizer, "decode", None)
    if not callable(decode_fn):
        return ""
    try:
        return str(decode_fn([token_id], skip_special_tokens=False))
    except Exception:
        return ""


def _build_sampling_params(
    *,
    temperature: Optional[float],
    top_p: Optional[float],
    max_tokens: Optional[int],
    stop: Any,
) -> SamplingParams:
    return SamplingParams(
        temperature=float(temperature) if temperature is not None else 1.0,
        top_p=float(top_p) if top_p is not None else 1.0,
        max_tokens=cast(int, max_tokens),
        stop=_extract_stop_sequences(stop),
    )


def _tokenize_text(prompt: str) -> list[int]:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="tokenizer not initialized")
    encode_fn = getattr(tokenizer, "encode", None)
    if not callable(encode_fn):
        raise HTTPException(
            status_code=500, detail="tokenizer.encode unavailable"
        )

    token_ids = encode_fn(prompt)
    if isinstance(token_ids, list):
        return [int(token_id) for token_id in token_ids]
    raise HTTPException(status_code=500, detail="failed to encode prompt")


def _chat_prompt_to_token_ids(request: ChatCompletionRequest) -> list[int]:
    if tokenizer is None:
        raise HTTPException(status_code=503, detail="tokenizer not initialized")

    if isinstance(request.messages, str):
        return _tokenize_text(request.messages)

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


def _ensure_runtime_ready() -> tuple[ContinuousBatchingEngine, StreamManager]:
    if engine is None:
        raise HTTPException(status_code=503, detail="engine not initialized")
    if stream_manager is None:
        raise HTTPException(
            status_code=503, detail="stream manager not initialized"
        )
    return engine, stream_manager


async def _wait_non_stream_result(
    *,
    request_id: str,
    prompt_token_ids: list[int],
    sampling_params: SamplingParams,
    raw_request: Request,
) -> tuple[str, dict[str, int], Optional[str]]:
    runtime_engine, _ = _ensure_runtime_ready()
    done_event = Event()
    token_texts: list[str] = []
    usage: dict[str, int] = {
        "prompt_tokens": len(prompt_token_ids),
        "completion_tokens": 0,
        "total_tokens": len(prompt_token_ids),
    }
    finish_reason: Optional[str] = None

    def on_token(output: RequestOutput) -> None:
        nonlocal usage, finish_reason
        token_texts.append(
            _decode_token_text(output.token_id, output.token_text)
        )
        if output.usage is not None:
            usage = output.usage
        if output.finished:
            finish_reason = output.finish_reason or "stop"
            done_event.set()

    runtime_engine.add_request(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        on_token=on_token,
    )

    while not done_event.is_set():
        if await raw_request.is_disconnected():
            runtime_engine.abort_request(request_id)
            raise HTTPException(status_code=499, detail="client disconnected")
        await asyncio.sleep(0.01)

    return "".join(token_texts), usage, finish_reason


def _next_stream_event(generator: Any) -> Optional[str]:
    try:
        return next(generator)
    except StopIteration:
        return None


def _make_completion_stream_chunk(
    *,
    request_id: str,
    created: int,
    model_name: str,
    token_text: str,
    finish_reason: Optional[str],
) -> str:
    resolved_finish_reason: Optional[Literal["stop", "length"]] = None
    if finish_reason == "stop":
        resolved_finish_reason = "stop"
    elif finish_reason == "length":
        resolved_finish_reason = "length"

    chunk = CompletionStreamResponse(
        id=f"cmpl-{request_id}",
        created=created,
        model=model_name,
        choices=[
            CompletionResponseStreamChoice(
                index=0,
                text=token_text,
                finish_reason=resolved_finish_reason,
            )
        ],
    )
    return f"data: {chunk.model_dump_json(exclude_none=True)}\n\n"


async def _completion_event_generator(
    *,
    request_id: str,
    created: int,
    model_name: str,
    stream: Any,
    raw_request: Request,
) -> Any:
    runtime_engine, _ = _ensure_runtime_ready()
    while True:
        if await raw_request.is_disconnected():
            runtime_engine.abort_request(request_id)
            return
        try:
            event = await asyncio.wait_for(
                asyncio.to_thread(_next_stream_event, stream),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            continue

        if event is None:
            return
        if event == "data: [DONE]\n\n":
            yield event
            return

        if not event.startswith("data: "):
            continue
        payload_text = event[len("data: ") :].strip()
        try:
            payload = json.loads(payload_text)
        except json.JSONDecodeError:
            continue

        choices = payload.get("choices", [])
        first_choice = choices[0] if choices else {}
        delta = first_choice.get("delta", {})
        token_text = delta.get("content", "") if isinstance(delta, dict) else ""
        finish_reason = first_choice.get("finish_reason")
        yield _make_completion_stream_chunk(
            request_id=request_id,
            created=created,
            model_name=model_name,
            token_text=token_text,
            finish_reason=finish_reason,
        )


async def _chat_event_generator(
    *, request_id: str, stream: Any, raw_request: Request
) -> Any:
    runtime_engine, _ = _ensure_runtime_ready()
    while True:
        if await raw_request.is_disconnected():
            runtime_engine.abort_request(request_id)
            return
        try:
            event = await asyncio.wait_for(
                asyncio.to_thread(_next_stream_event, stream),
                timeout=0.1,
            )
        except asyncio.TimeoutError:
            continue

        if event is None:
            return
        yield event
        if event == "data: [DONE]\n\n":
            return


async def _engine_loop() -> None:
    while (
        _engine_shutdown_event is not None
        and not _engine_shutdown_event.is_set()
    ):
        if engine is None:
            await asyncio.sleep(0.01)
            continue

        if engine.has_pending_requests():
            if _decode_watchdog is not None:
                _decode_watchdog.activate()
            _ = engine.step()
            if _decode_watchdog is not None:
                _decode_watchdog.feed()
            await asyncio.sleep(0)
            continue

        if _decode_watchdog is not None:
            _decode_watchdog.deactivate()
        await asyncio.sleep(0.005)


def _ensure_engine_loop_running() -> None:
    global _engine_task
    global _engine_shutdown_event

    if engine is None:
        return

    if _engine_shutdown_event is None or _engine_shutdown_event.is_set():
        _engine_shutdown_event = asyncio.Event()

    if _engine_task is None or _engine_task.done():
        _engine_task = asyncio.create_task(_engine_loop())


async def _initialize_model() -> None:
    global engine
    global _health_state
    global runtime_max_seq_length
    global stream_manager
    global tokenizer
    global model_name_global
    global _startup_watchdog
    global _decode_watchdog
    global _watchdog_config

    args = _startup_args
    if args is None or engine is not None:
        return

    _health_state.set_starting()
    _startup_watchdog = None
    _decode_watchdog = None
    _watchdog_config = None

    startup_timeout = getattr(args, "startup_timeout", None)
    decode_step_timeout = getattr(args, "decode_step_timeout", None)
    if startup_timeout is not None or decode_step_timeout is not None:
        _watchdog_config = watchdog_module.WatchdogConfig(
            startup_timeout=startup_timeout,
            decode_step_timeout=decode_step_timeout,
            enable_pyspy_dump=getattr(args, "enable_pyspy_dump", False),
        )
        _startup_watchdog = watchdog_module.start_startup_watchdog(
            _health_state,
            _watchdog_config,
            is_ready=lambda: engine is not None,
        )

    try:
        from transformers import AutoTokenizer

        model_name_global = args.model
        tokenizer = AutoTokenizer.from_pretrained(
            args.model,
            trust_remote_code=True,
        )

        moe_config = {
            "offload_path": os.path.join(args.offload_dir, args.model),
            "device_memory_ratio": args.device_memory_ratio,
        }
        if args.enable_prefix_caching:
            moe_config["enable_prefix_caching"] = True

        moe_module = importlib.import_module("moe_infinity")
        moe_ctor = getattr(moe_module, "MoE", None)
        if moe_ctor is None:
            raise RuntimeError("MoE is unavailable in this environment")

        moe_model = moe_ctor(args.model, moe_config)
        engine_config = _build_engine_config(args=args, model=moe_model.model)

        initialized_engine = ContinuousBatchingEngine(
            model=moe_model.model,
            engine=moe_model.engine,
            config=engine_config,
            tokenizer=tokenizer,
        )
        if stream_manager is None:
            stream_manager = StreamManager()

        engine = initialized_engine
        configured_max_seq_length = engine_config.get("max_seq_length")
        if isinstance(configured_max_seq_length, int):
            runtime_max_seq_length = configured_max_seq_length

        if _watchdog_config is not None:
            _decode_watchdog = watchdog_module.start_decode_watchdog(
                _health_state,
                _watchdog_config,
            )

        _ensure_engine_loop_running()
    finally:
        if _startup_watchdog is not None:
            _startup_watchdog.cancel()

    _health_state.set_healthy()


@app.on_event("startup")
async def startup_event() -> None:
    global _model_init_task
    global _engine_task
    global stream_manager

    if stream_manager is None:
        stream_manager = StreamManager()

    _ = _ensure_cp_middleware_initialized()

    if _startup_args is not None and engine is None:
        if _model_init_task is None or _model_init_task.done():
            _model_init_task = asyncio.create_task(_initialize_model())
        return

    _ensure_engine_loop_running()


@app.on_event("shutdown")
async def shutdown_event() -> None:
    global _engine_task
    global _model_init_task
    global _decode_watchdog

    if _model_init_task is not None:
        if not _model_init_task.done():
            _model_init_task.cancel()
        with suppress(asyncio.CancelledError):
            await _model_init_task
        _model_init_task = None

    if _engine_shutdown_event is not None:
        _engine_shutdown_event.set()

    if _engine_task is not None:
        await _engine_task
        _engine_task = None

    if _decode_watchdog is not None:
        _decode_watchdog.deactivate()
        _decode_watchdog.stop()
        _decode_watchdog.join(timeout=1.0)
        _decode_watchdog = None

    if engine is not None:
        shutdown_fn = getattr(engine, "shutdown", None)
        if callable(shutdown_fn):
            shutdown_fn()


@app.get("/health")
async def health() -> Response:
    status_dict = _health_state.get_status_dict()
    is_healthy = _health_state.is_healthy()
    status_code = 200 if is_healthy else 503
    return JSONResponse(content=status_dict, status_code=status_code)


@app.post("/contextpilot/toggle")
async def contextpilot_toggle(payload: dict[str, Any]) -> JSONResponse:
    enabled = payload.get("enabled")
    if not isinstance(enabled, bool):
        raise HTTPException(
            status_code=400,
            detail="'enabled' must be a boolean",
        )

    with _contextpilot_state_lock:
        global _contextpilot_enabled
        _contextpilot_enabled = enabled

    if enabled:
        _ = _ensure_cp_middleware_initialized()

    return JSONResponse(
        content={"enabled": enabled},
        status_code=200,
    )


@app.post("/contextpilot/inject-fault")
async def contextpilot_inject_fault(payload: dict[str, Any]) -> JSONResponse:
    with _contextpilot_state_lock:
        if not _contextpilot_debug:
            raise HTTPException(
                status_code=403,
                detail="ContextPilot debug endpoints are disabled",
            )

    fault = payload.get("fault")
    if fault not in {"none", "reorder_exception"}:
        raise HTTPException(
            status_code=400,
            detail="'fault' must be one of: none, reorder_exception",
        )

    duration_s = payload.get("duration_s", 0)
    if not isinstance(duration_s, int) or duration_s < 0:
        raise HTTPException(
            status_code=400,
            detail="'duration_s' must be a non-negative integer",
        )

    with _contextpilot_state_lock:
        global _contextpilot_fault
        _contextpilot_fault = fault

    return JSONResponse(
        content={
            "fault": fault,
            "duration_s": duration_s,
        },
        status_code=200,
    )


@app.get("/contextpilot/status")
async def contextpilot_status() -> JSONResponse:
    with _contextpilot_state_lock:
        enabled = _contextpilot_enabled
        debug = _contextpilot_debug
        fault = _contextpilot_fault
        fallback_count = _contextpilot_fallback_count
        last_fallback_count = _contextpilot_last_fallback_count

    return JSONResponse(
        content={
            "enabled": enabled,
            "debug": debug,
            "fault": fault,
            "fallback_count": fallback_count,
            "last_fallback_count": last_fallback_count,
            "env_enabled": os.environ.get("CONTEXTPILOT_ENABLED", "1") != "0",
        },
        status_code=200,
    )


@app.post("/v1/completions")
async def completion(request: CompletionRequest, raw_request: Request):
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service is starting. Please retry shortly.",
            error_type="server_error",
            code="service_starting",
        )

    runtime_engine, runtime_stream_manager = _ensure_runtime_ready()
    created_time = int(time.monotonic())
    model_name = request.model or model_name_global or "unknown"

    try:
        validate_required_params(request.max_tokens)
        validate_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except InvalidRequestError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
            param=exc.param,
        )

    try:
        prompt_is_tokens, prompts = parse_prompt_format(request.prompt)
    except ValueError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
        )

    sampling_params = _build_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )

    if request.stream:
        if len(prompts) != 1:
            return create_error_response(
                status_code=400,
                message="streaming completion only supports a single prompt",
                error_type="invalid_request_error",
                code="invalid_request_error",
            )

        request_id = random_uuid()
        prompt = prompts[0]
        if prompt_is_tokens:
            prompt_token_ids = [int(token_id) for token_id in prompt]
        else:
            processed_prompt = _process_completion_prompt_with_contextpilot(
                str(prompt),
                request_id=request_id,
            )
            prompt_token_ids = _tokenize_text(processed_prompt)

        try:
            validate_context_length(
                len(prompt_token_ids),
                cast(int, request.max_tokens),
                runtime_max_seq_length,
            )
        except ContextLengthExceededError as exc:
            return create_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )

        stream = runtime_stream_manager.create_stream(
            request_id=request_id,
            model=model_name,
        )

        def on_stream_token(output: RequestOutput) -> None:
            runtime_stream_manager.push_token(
                request_id=request_id,
                token_text=_decode_token_text(
                    output.token_id, output.token_text
                ),
                finished=output.finished,
                finish_reason=output.finish_reason,
            )

        runtime_engine.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            on_token=on_stream_token,
        )

        return StreamingResponse(
            _completion_event_generator(
                request_id=request_id,
                created=created_time,
                model_name=model_name,
                stream=stream,
                raw_request=raw_request,
            ),
            media_type="text/event-stream",
        )

    choices: list[CompletionResponseChoice] = []
    usage_prompt_tokens = 0
    usage_completion_tokens = 0
    for index, prompt in enumerate(prompts):
        request_id = random_uuid()
        if prompt_is_tokens:
            prompt_token_ids = [int(token_id) for token_id in prompt]
        else:
            processed_prompt = _process_completion_prompt_with_contextpilot(
                str(prompt),
                request_id=request_id,
            )
            prompt_token_ids = _tokenize_text(processed_prompt)

        try:
            validate_context_length(
                len(prompt_token_ids),
                cast(int, request.max_tokens),
                runtime_max_seq_length,
            )
        except ContextLengthExceededError as exc:
            return create_error_response(
                status_code=400,
                message=str(exc),
                error_type="invalid_request_error",
                code="context_length_exceeded",
            )

        (
            output_text,
            usage,
            result_finish_reason,
        ) = await _wait_non_stream_result(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            raw_request=raw_request,
        )

        choices.append(
            CompletionResponseChoice(
                index=index,
                text=output_text,
                logprobs=None,
                finish_reason=cast(
                    Literal["stop", "length"], result_finish_reason or "stop"
                ),
            )
        )
        usage_prompt_tokens += usage.get("prompt_tokens", 0)
        usage_completion_tokens += usage.get("completion_tokens", 0)

    return CompletionResponse(
        id=f"cmpl-{random_uuid()}",
        created=created_time,
        model=model_name,
        choices=choices,
        usage=UsageInfo(
            prompt_tokens=usage_prompt_tokens,
            completion_tokens=usage_completion_tokens,
            total_tokens=usage_prompt_tokens + usage_completion_tokens,
        ),
    )


@app.post("/v1/chat/completions")
async def chat_completion(request: ChatCompletionRequest, raw_request: Request):
    if engine is None:
        return create_error_response(
            status_code=503,
            message="Service is starting. Please retry shortly.",
            error_type="server_error",
            code="service_starting",
        )

    runtime_engine, runtime_stream_manager = _ensure_runtime_ready()
    created_time = int(time.monotonic())
    model_name = request.model or model_name_global or "unknown"
    request_id = random_uuid()

    try:
        validate_required_params(request.max_tokens)
        validate_sampling_params(
            temperature=request.temperature,
            top_p=request.top_p,
            max_tokens=request.max_tokens,
        )
    except InvalidRequestError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="invalid_request_error",
            param=exc.param,
        )

    sampling_params = _build_sampling_params(
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        stop=request.stop,
    )
    processed_messages = _process_chat_messages_with_contextpilot(
        request.messages,
        request_id=request_id,
    )
    original_messages = request.messages
    request.messages = processed_messages
    try:
        prompt_token_ids = _chat_prompt_to_token_ids(request)
    finally:
        request.messages = original_messages

    try:
        validate_context_length(
            len(prompt_token_ids),
            cast(int, request.max_tokens),
            runtime_max_seq_length,
        )
    except ContextLengthExceededError as exc:
        return create_error_response(
            status_code=400,
            message=str(exc),
            error_type="invalid_request_error",
            code="context_length_exceeded",
        )

    if request.stream:
        stream = runtime_stream_manager.create_stream(
            request_id=request_id,
            model=model_name,
        )

        def on_stream_token(output: RequestOutput) -> None:
            runtime_stream_manager.push_token(
                request_id=request_id,
                token_text=_decode_token_text(
                    output.token_id, output.token_text
                ),
                finished=output.finished,
                finish_reason=output.finish_reason,
            )

        runtime_engine.add_request(
            request_id=request_id,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            on_token=on_stream_token,
        )
        return StreamingResponse(
            _chat_event_generator(
                request_id=request_id,
                stream=stream,
                raw_request=raw_request,
            ),
            media_type="text/event-stream",
        )

    output_text, usage, chat_finish_reason = await _wait_non_stream_result(
        request_id=request_id,
        prompt_token_ids=prompt_token_ids,
        sampling_params=sampling_params,
        raw_request=raw_request,
    )

    return ChatCompletionResponse(
        id=f"chatcmpl-{request_id}",
        created=created_time,
        model=model_name,
        choices=[
            ChatCompletionResponseChoice(
                index=0,
                message=ChatMessage(role="assistant", content=output_text),
                finish_reason=cast(
                    Literal["stop", "length"], chat_finish_reason or "stop"
                ),
            )
        ],
        usage=UsageInfo(
            prompt_tokens=usage.get("prompt_tokens", len(prompt_token_ids)),
            completion_tokens=usage.get("completion_tokens", 0),
            total_tokens=usage.get("total_tokens", len(prompt_token_ids)),
        ),
    )


def _resolve_int_attr(config: object, *names: str) -> Optional[int]:
    for name in names:
        value = getattr(config, name, None)
        if isinstance(value, int):
            return value
    return None


def _resolve_dtype(model: object) -> str:
    model_config = getattr(model, "config", None)
    if model_config is not None:
        torch_dtype = getattr(model_config, "torch_dtype", None)
        if isinstance(torch_dtype, str):
            return torch_dtype.replace("torch.", "")
        if torch_dtype is not None:
            return str(torch_dtype).replace("torch.", "")
    return "float16"


def _build_engine_config(
    args: argparse.Namespace, model: object
) -> dict[str, object]:
    model_config = getattr(model, "config", None)
    if model_config is None:
        raise RuntimeError(
            "model config is required to initialize serving engine"
        )

    num_layers = _resolve_int_attr(
        model_config,
        "num_hidden_layers",
        "num_layers",
        "n_layer",
    )
    num_attention_heads = _resolve_int_attr(
        model_config,
        "num_attention_heads",
        "n_head",
    )
    num_kv_heads = _resolve_int_attr(
        model_config,
        "num_key_value_heads",
        "num_kv_heads",
        "n_head_kv",
    )
    hidden_size = _resolve_int_attr(model_config, "hidden_size", "n_embd")
    max_seq_length = _resolve_int_attr(
        model_config,
        "max_position_embeddings",
        "max_seq_len",
        "max_sequence_length",
        "n_positions",
        "model_max_length",
    )
    head_dim = _resolve_int_attr(model_config, "head_dim")

    if num_layers is None:
        raise RuntimeError("unable to resolve model num_layers")
    if num_attention_heads is None:
        raise RuntimeError("unable to resolve model num_attention_heads")
    if num_kv_heads is None:
        num_kv_heads = num_attention_heads
    if head_dim is None:
        if hidden_size is None:
            raise RuntimeError("unable to resolve model hidden_size/head_dim")
        head_dim = hidden_size // max(1, num_attention_heads)

    eos_token_id: Optional[int] = _resolve_int_attr(
        model_config, "eos_token_id"
    )
    if eos_token_id is None:
        config_eos = getattr(model_config, "eos_token_id", None)
        if (
            isinstance(config_eos, list)
            and config_eos
            and isinstance(config_eos[0], int)
        ):
            eos_token_id = config_eos[0]

    config: dict[str, object] = {
        "device_memory_ratio": args.device_memory_ratio,
        "kv_cache_ratio": args.kv_cache_ratio,
        "max_batch_size": args.max_batch_size,
        "max_tokens_per_step": 2048,
        "max_seq_length": max_seq_length or 4096,
        "block_size": 16,
        "num_layers": num_layers,
        "num_kv_heads": num_kv_heads,
        "head_dim": head_dim,
        "dtype": _resolve_dtype(model),
    }
    if eos_token_id is not None:
        config["eos_token_id"] = eos_token_id
    if args.enable_prefix_caching:
        config["enable_prefix_caching"] = True
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MoE-Infinity OpenAI-Compatible RESTful API server v2."
    )
    parser.add_argument("--host", type=str, default="0.0.0.0", help="host name")
    parser.add_argument("--port", type=int, default=8000, help="port number")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--offload-dir", type=str, required=True)
    parser.add_argument("--device-memory-ratio", type=float, default=0.75)
    parser.add_argument("--kv-cache-ratio", type=float, default=0.25)
    parser.add_argument("--max-batch-size", type=int, default=32)
    parser.add_argument("--enable-prefix-caching", action="store_true")
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=None,
        help="Startup watchdog timeout in seconds (disabled by default)",
    )
    parser.add_argument(
        "--decode-step-timeout",
        type=float,
        default=None,
        help="Decode step watchdog timeout in seconds (disabled by default)",
    )
    parser.add_argument(
        "--enable-pyspy-dump",
        action="store_true",
        help="Enable py-spy stack dump on watchdog timeout (requires py-spy installed)",
    )
    parser.add_argument(
        "--enable-contextpilot",
        action="store_true",
        default=False,
        help="Enable ContextPilot middleware",
    )
    parser.add_argument(
        "--contextpilot-debug",
        action="store_true",
        default=False,
        help="Enable CP debug endpoints (inject-fault)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    with _contextpilot_state_lock:
        _contextpilot_enabled = _resolve_contextpilot_enabled(
            args.enable_contextpilot
        )
        _contextpilot_debug = bool(args.contextpilot_debug)
        _contextpilot_fault = "none"
    _startup_args = args

    uvicorn.run(
        app,
        host=args.host,
        port=args.port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE,
    )
