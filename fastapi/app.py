"""FastAPI demo server for nano-vllm VoxCPM.

This module exposes a small HTTP API around :class:`~nanovllm_voxcpm.models.voxcpm.server.AsyncVoxCPMServerPool`.

The endpoints are intentionally lightweight (no auth, no persistence) and are
meant for local demos / prototyping.

Notes on audio formats:
    - ``/add_prompt`` expects an *entire audio file* encoded as base64.
      It is *not* a data URI.
    - ``/generate`` returns a streaming response of raw waveform bytes.
      The stream is interpreted as little-endian ``float32`` mono PCM at 44.1 kHz
      (see endpoint docstring for an exact parsing example).
"""

from __future__ import annotations

import os
import base64
from contextlib import asynccontextmanager
from typing import AsyncIterator, Literal, Protocol, TypeVar

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from nanovllm_voxcpm.models.voxcpm.config import LoRAConfig
from nanovllm_voxcpm.models.voxcpm.server import AsyncVoxCPMServerPool

class HealthResponse(BaseModel):
    """Health check response."""

    status: Literal["ok"] = "ok"


class StatusOKResponse(BaseModel):
    """Simple ok response used by mutating endpoints."""

    status: Literal["ok"] = "ok"


class LoadLoRARequest(BaseModel):
    """Request body for loading LoRA weights."""

    lora_path: str = Field(..., description="Filesystem path to the LoRA weights to load.")


class LoadLoRAResponse(BaseModel):
    """Response for a successful LoRA load."""

    status: Literal["ok"] = "ok"
    loaded_keys: int = Field(..., ge=0, description="Number of parameter keys loaded from the LoRA file.")
    skipped_keys: int = Field(..., ge=0, description="Number of parameter keys skipped (mismatch / missing).")


class SetLoRAEnabledRequest(BaseModel):
    """Request body for toggling LoRA enablement."""

    enabled: bool = Field(..., description="Whether LoRA is enabled for generation.")


class SetLoRAEnabledResponse(BaseModel):
    """Response for toggling LoRA enablement."""

    status: Literal["ok"] = "ok"
    lora_enabled: bool = Field(..., description="Current LoRA enable state.")


class AddPromptRequest(BaseModel):
    """Request body for adding a prompt.

    The prompt consists of an audio file (base64-encoded bytes) and a text
    transcript used by the model.
    """

    wav_base64: str = Field(
        ...,
        description=(
            "Base64-encoded audio file bytes (entire file contents). "
            "Do not include a 'data:audio/...' prefix."
        ),
    )
    wav_format: str = Field(
        ...,
        description=(
            "Audio container format for decoding (e.g. 'wav', 'flac', 'mp3'). "
            "This is passed through to torchaudio."
        ),
        examples=["wav"],
    )
    prompt_text: str = Field(..., description="Transcript / prompt text associated with the audio.")


class AddPromptResponse(BaseModel):
    """Response containing the prompt id."""

    prompt_id: str = Field(..., description="ID to reference this prompt in /generate.")


class RemovePromptRequest(BaseModel):
    """Request body for removing a previously added prompt."""

    prompt_id: str = Field(..., description="Prompt id returned by /add_prompt.")


class GenerateRequest(BaseModel):
    """Request body for streaming generation.

    Provide ``prompt_id`` if you want to condition generation on a previously
    added prompt.
    """

    target_text: str = Field(..., description="Text to synthesize.")
    prompt_id: str | None = Field(
        None,
        description="Optional prompt id from /add_prompt. If omitted, generation is zero-shot.",
    )
    max_generate_length: int = Field(2000, ge=1, description="Maximum number of model generation steps.")
    temperature: float = Field(1.0, ge=0.0, description="Sampling temperature.")
    cfg_value: float = Field(1.5, ge=0.0, description="Classifier-free guidance scale.")


# ==================== Configuration ====================
MODEL_PATH: str = os.path.expanduser("~/VoxCPM1.5")
LORA_PATH: str | None = None

# LoRA configuration (set to None to disable LoRA structure)
# LORA_CONFIG = LoRAConfig(
#     enable_lm=True,
#     enable_dit=True,
#     enable_proj=False,
#     r=32,
#     alpha=16.0,
#     target_modules_lm=["q_proj", "k_proj", "v_proj", "o_proj"],
#     target_modules_dit=["q_proj", "k_proj", "v_proj", "o_proj"],
# )
# If LoRA is not needed, set to None:
LORA_CONFIG: LoRAConfig | None = None
# ================================================

def _get_server(request: Request) -> AsyncVoxCPMServerPool:
    """FastAPI dependency that returns the initialized server pool."""

    server = getattr(request.app.state, "server", None)
    if server is None:
        raise HTTPException(status_code=503, detail="Model server not ready")
    return server


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Create and tear down the model server pool."""

    app.state.server = AsyncVoxCPMServerPool(
        model_path=MODEL_PATH,
        max_num_batched_tokens=8192,
        max_num_seqs=16,
        max_model_len=4096,
        gpu_memory_utilization=0.95,
        enforce_eager=False,
        devices=[0],
        lora_config=LORA_CONFIG,  # Add LoRA config
    )
    await app.state.server.wait_for_ready()  # Wait for model to load first

    # Then load LoRA weights (optional)
    if LORA_PATH:
        await app.state.server.load_lora(LORA_PATH)
        await app.state.server.set_lora_enabled(True)
    yield
    await app.state.server.stop()
    delattr(app.state, "server")


app = FastAPI(
    title="nano-vllm VoxCPM Demo",
    description="Minimal FastAPI wrapper around VoxCPM for prompt management and streaming generation.",
    version="0.1.0",
    openapi_tags=[
        {"name": "health", "description": "Basic liveness checks."},
        {"name": "lora", "description": "LoRA weight loading and enable/disable."},
        {"name": "prompts", "description": "Prompt pool management."},
        {"name": "generation", "description": "Streaming text-to-audio generation."},
    ],
    lifespan=lifespan,
)


_TModel = TypeVar("_TModel", bound=BaseModel)


def _validate_model(model_cls: type[_TModel], payload: object) -> _TModel:
    """Validate dict-like payloads across Pydantic v1/v2."""

    if hasattr(model_cls, "model_validate"):
        # Pydantic v2
        return model_cls.model_validate(payload)  # type: ignore[attr-defined]
    # Pydantic v1
    return model_cls.parse_obj(payload)  # type: ignore[attr-defined]


@app.get("/health", response_model=HealthResponse, tags=["health"])
async def health() -> HealthResponse:
    """Health check.

    Returns ``{"status": "ok"}`` when the server process is responsive.
    """

    return HealthResponse()


# ==================== LoRA Management API ====================

@app.post("/lora/load", response_model=LoadLoRAResponse, tags=["lora"])
async def load_lora(
    request: LoadLoRARequest,
    server: AsyncVoxCPMServerPool = Depends(_get_server),
) -> LoadLoRAResponse:
    """Load LoRA weights from disk."""

    result = await server.load_lora(request.lora_path)
    # The underlying server returns a TypedDict-like payload.
    return _validate_model(LoadLoRAResponse, result)


@app.post("/lora/set_enabled", response_model=SetLoRAEnabledResponse, tags=["lora"])
async def set_lora_enabled(
    request: SetLoRAEnabledRequest,
    server: AsyncVoxCPMServerPool = Depends(_get_server),
) -> SetLoRAEnabledResponse:
    """Enable/disable LoRA."""

    result = await server.set_lora_enabled(request.enabled)
    return _validate_model(SetLoRAEnabledResponse, result)


@app.post("/lora/reset", response_model=StatusOKResponse, tags=["lora"])
async def reset_lora(server: AsyncVoxCPMServerPool = Depends(_get_server)) -> StatusOKResponse:
    """Reset (unload) LoRA weights."""

    result = await server.reset_lora()
    # The server returns {"status": "ok"}; validate to keep OpenAPI stable.
    return _validate_model(StatusOKResponse, result)


# ==================== Original API ====================

@app.post("/add_prompt", response_model=AddPromptResponse, tags=["prompts"])
async def add_prompt(
    request: AddPromptRequest,
    server: AsyncVoxCPMServerPool = Depends(_get_server),
) -> AddPromptResponse:
    """Register an audio+text prompt and return its id."""

    try:
        wav = base64.b64decode(request.wav_base64)
    except Exception as e:  # pragma: no cover
        raise HTTPException(status_code=400, detail=f"Invalid base64 in wav_base64: {e}") from e

    prompt_id = await server.add_prompt(wav, request.wav_format, request.prompt_text)
    return AddPromptResponse(prompt_id=prompt_id)


@app.post("/remove_prompt", response_model=StatusOKResponse, tags=["prompts"])
async def remove_prompt(
    request: RemovePromptRequest,
    server: AsyncVoxCPMServerPool = Depends(_get_server),
) -> StatusOKResponse:
    """Remove a stored prompt from the in-memory prompt pool."""

    await server.remove_prompt(request.prompt_id)
    return StatusOKResponse()


class _SupportsToBytes(Protocol):
    def tobytes(self) -> bytes:  # pragma: no cover
        ...


async def _chunks_to_bytes(gen: AsyncIterator[_SupportsToBytes]) -> AsyncIterator[bytes]:
    """Convert an async iterator of "array-like" chunks into bytes."""

    async for data in gen:
        yield data.tobytes()


@app.post(
    "/generate",
    response_class=StreamingResponse,
    tags=["generation"],
    responses={
        200: {
            "description": (
                "Raw audio stream. The response body is a byte stream containing "
                "contiguous float32 PCM samples (mono).\n\n"
                "Parsing example (Python):\n"
                "- Read all bytes\n"
                "- wav = np.frombuffer(data, dtype=np.float32)\n"
                "- sf.write('out.wav', wav, 44100)"
            ),
            "content": {"audio/raw": {"schema": {"type": "string", "format": "binary"}}},
            "headers": {
                "X-Audio-Sample-Rate": {
                    "description": "Sample rate (Hz) for the returned waveform.",
                    "schema": {"type": "integer", "example": 44100},
                },
                "X-Audio-Channels": {
                    "description": "Number of audio channels.",
                    "schema": {"type": "integer", "example": 1},
                },
                "X-Audio-DType": {
                    "description": "Sample dtype for the raw byte stream.",
                    "schema": {"type": "string", "example": "float32"},
                },
            },
        }
    },
)
async def generate(
    request: GenerateRequest,
    server: AsyncVoxCPMServerPool = Depends(_get_server),
) -> StreamingResponse:
    """Stream generated audio bytes.

    Response format
    ---------------
    - The HTTP body is streamed (chunked transfer encoding).
    - ``Content-Type`` is ``audio/raw``.
    - The stream consists of contiguous ``float32`` PCM samples (mono).

    How to parse (Python)
    ---------------------
    The demo client reads the full response into memory and writes a WAV file
    like this:

    ```python
    import numpy as np
    import soundfile as sf

    data = await response.content.read()  # bytes
    wav = np.frombuffer(data, dtype=np.float32)
    sf.write("out.wav", wav, 44100)
    ```

    If you want true streaming playback, you can incrementally buffer chunks
    from the response and append them to a float32 ring buffer.
    """

    return StreamingResponse(
        _chunks_to_bytes(
            server.generate(
                target_text=request.target_text,
                prompt_latents=None,
                prompt_text="",
                prompt_id=request.prompt_id,
                max_generate_length=request.max_generate_length,
                temperature=request.temperature,
                cfg_value=request.cfg_value,
            )
        ),
        media_type="audio/raw",
        headers={
            # These are primarily for documentation and interoperability.
            # The current demo returns 44.1 kHz mono float32 samples.
            "X-Audio-Sample-Rate": "44100",
            "X-Audio-Channels": "1",
            "X-Audio-DType": "float32",
        },
    )
