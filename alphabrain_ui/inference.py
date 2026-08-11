"""Validated, UI-managed inference against an AlphaBrain model deployment."""

from __future__ import annotations

import asyncio
import hashlib
import json
import math
import time
import uuid
import warnings
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import websockets.asyncio.client
from fastapi import UploadFile
from PIL import Image, UnidentifiedImageError

from deployment.model_server.tools import msgpack_numpy

MAX_IMAGE_BYTES = 10 * 1024 * 1024
MAX_IMAGE_COUNT = 4
MAX_BATCH_SIZE = 8
MAX_INSTRUCTION_LENGTH = 4096
MAX_STATE_ELEMENTS = 16_384
MAX_OUTPUT_BYTES = 32 * 1024 * 1024
INFERENCE_TIMEOUT_SECONDS = 120.0
ALLOWED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp"}


class InferenceInputError(ValueError):
    def __init__(self, code: str, *, status_code: int = 422):
        super().__init__(code)
        self.code = code
        self.status_code = status_code


class InferenceTransportError(RuntimeError):
    def __init__(self, code: str, detail: str = ""):
        super().__init__(detail or code)
        self.code = code
        self.detail = detail


@dataclass(frozen=True)
class UploadedImage:
    filename: str
    content_type: str
    data: bytes
    width: int
    height: int
    sha256: str
    array: np.ndarray

    def summary(self) -> dict[str, Any]:
        return {
            "filename": self.filename,
            "content_type": self.content_type,
            "size_bytes": len(self.data),
            "width": self.width,
            "height": self.height,
            "sha256": self.sha256,
        }


@dataclass(frozen=True)
class DecodedInferenceRequest:
    request_id: str
    payload: dict[str, Any]
    summary: dict[str, Any]
    images: tuple[UploadedImage, ...]
    batch_size: int


def _parse_json_or_text(value: str, *, code: str) -> Any:
    value = value.strip()
    if not value:
        raise InferenceInputError(code)
    if value[:1] not in {"[", "{", '"'}:
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError as exc:
        raise InferenceInputError(code) from exc


def _parse_instructions(value: str) -> list[str]:
    parsed = _parse_json_or_text(value, code="invalid_instructions")
    if isinstance(parsed, str):
        parsed = [parsed]
    if not isinstance(parsed, list) or not 1 <= len(parsed) <= MAX_BATCH_SIZE:
        raise InferenceInputError("invalid_batch_size")
    result: list[str] = []
    for item in parsed:
        if not isinstance(item, str):
            raise InferenceInputError("invalid_instructions")
        item = item.strip()
        if not item or len(item) > MAX_INSTRUCTION_LENGTH or any(ord(char) == 0 for char in item):
            raise InferenceInputError("invalid_instruction")
        result.append(item)
    return result


def _parse_image_counts(value: str | None, *, batch_size: int, image_count: int) -> list[int] | None:
    if value is None or not value.strip():
        return None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InferenceInputError("invalid_image_counts") from exc
    if (
        not isinstance(parsed, list)
        or len(parsed) != batch_size
        or any(isinstance(item, bool) or not isinstance(item, int) or item < 0 for item in parsed)
        or sum(parsed) != image_count
    ):
        raise InferenceInputError("invalid_image_counts")
    return parsed


def _parse_states(value: str | None, *, batch_size: int) -> tuple[list[Any] | None, dict[str, Any] | None]:
    if value is None or not value.strip():
        return None, None
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise InferenceInputError("invalid_states") from exc
    try:
        array = np.asarray(parsed, dtype=np.float32)
    except (TypeError, ValueError, OverflowError) as exc:
        raise InferenceInputError("invalid_states") from exc
    if array.ndim == 0 or array.ndim > 3 or array.size > MAX_STATE_ELEMENTS:
        raise InferenceInputError("invalid_states")
    if not np.isfinite(array).all():
        raise InferenceInputError("invalid_states")
    if batch_size == 1:
        if array.ndim == 1:
            array = array[np.newaxis, :]
        elif array.shape[0] != 1:
            array = array[np.newaxis, ...]
    elif array.shape[0] != batch_size:
        raise InferenceInputError("states_batch_mismatch")
    return array.tolist(), {
        "shape": list(array.shape),
        "dtype": "float32",
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


async def _decode_image(upload: UploadFile) -> UploadedImage:
    content_type = (upload.content_type or "").lower()
    if content_type not in ALLOWED_IMAGE_TYPES:
        raise InferenceInputError("unsupported_image_type")
    data = await upload.read(MAX_IMAGE_BYTES + 1)
    if not data:
        raise InferenceInputError("empty_image")
    if len(data) > MAX_IMAGE_BYTES:
        raise InferenceInputError("image_too_large", status_code=413)
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(data)) as image:
                image.verify()
            with Image.open(BytesIO(data)) as image:
                rgb = image.convert("RGB")
                width, height = rgb.size
                if width < 1 or height < 1 or width * height > 40_000_000:
                    raise InferenceInputError("image_dimensions_out_of_range")
                array = np.asarray(rgb, dtype=np.uint8).copy()
    except InferenceInputError:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombWarning) as exc:
        raise InferenceInputError("invalid_image") from exc
    return UploadedImage(
        filename=Path(upload.filename or "image").name[:255],
        content_type=content_type,
        data=data,
        width=width,
        height=height,
        sha256=hashlib.sha256(data).hexdigest(),
        array=array,
    )


async def decode_inference_request(
    *,
    instructions: str,
    states: str | None,
    image_counts: str | None,
    images: Sequence[UploadFile],
) -> DecodedInferenceRequest:
    instruction_values = _parse_instructions(instructions)
    if len(images) > MAX_IMAGE_COUNT:
        raise InferenceInputError("too_many_images", status_code=413)
    decoded_images = tuple([await _decode_image(image) for image in images])
    counts = _parse_image_counts(
        image_counts,
        batch_size=len(instruction_values),
        image_count=len(decoded_images),
    )
    if counts is None:
        # The normal Playground interaction broadcasts the same camera views
        # across a small instruction batch. Callers needing distinct views can
        # provide explicit flattened image counts.
        batch_images = [[image.array for image in decoded_images] for _ in instruction_values]
        summary_counts = [len(decoded_images)] * len(instruction_values)
        image_mode = "broadcast"
    else:
        batch_images = []
        offset = 0
        for count in counts:
            batch_images.append([image.array for image in decoded_images[offset : offset + count]])
            offset += count
        summary_counts = counts
        image_mode = "partitioned"
    state_values, state_summary = _parse_states(states, batch_size=len(instruction_values))
    payload: dict[str, Any] = {
        "batch_images": batch_images,
        "instructions": instruction_values,
    }
    if state_values is not None:
        payload["states"] = state_values
    request_id = str(uuid.uuid4())
    summary = {
        "schema_version": "inference-request-summary-v1",
        "request_id": request_id,
        "batch_size": len(instruction_values),
        "instructions": instruction_values,
        "image_mode": image_mode,
        "image_counts": summary_counts,
        "images": [image.summary() for image in decoded_images],
        "state": state_summary,
    }
    return DecodedInferenceRequest(
        request_id=request_id,
        payload=payload,
        summary=summary,
        images=decoded_images,
        batch_size=len(instruction_values),
    )


def json_safe(value: Any, *, _depth: int = 0) -> Any:
    if _depth > 32:
        raise InferenceTransportError("inference_output_too_deep")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.generic):
        return json_safe(value.item(), _depth=_depth + 1)
    if isinstance(value, np.ndarray):
        if value.size > 1_000_000:
            raise InferenceTransportError("inference_output_too_large")
        return json_safe(value.tolist(), _depth=_depth + 1)
    if isinstance(value, dict):
        return {str(key): json_safe(item, _depth=_depth + 1) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(item, _depth=_depth + 1) for item in value]
    raise InferenceTransportError("unsupported_inference_output", type(value).__name__)


async def infer_via_websocket(
    *,
    port: int,
    controller_key: str,
    request: DecodedInferenceRequest,
    timeout: float = INFERENCE_TIMEOUT_SECONDS,
) -> tuple[dict[str, Any], dict[str, Any], float]:
    uri = f"ws://127.0.0.1:{int(port)}"
    started = time.perf_counter()

    async def exchange() -> tuple[Any, Any]:
        async with websockets.asyncio.client.connect(
            uri,
            additional_headers={"Authorization": f"Api-Key {controller_key}"},
            compression=None,
            max_size=MAX_OUTPUT_BYTES,
            open_timeout=min(15.0, timeout),
            close_timeout=5.0,
            proxy=None,
        ) as websocket:
            metadata_raw = await websocket.recv()
            if isinstance(metadata_raw, str):
                raise InferenceTransportError("invalid_deployment_metadata")
            metadata_value = json_safe(msgpack_numpy.unpackb(metadata_raw))
            message = {
                "type": "infer",
                "request_id": request.request_id,
                "payload": request.payload,
            }
            await websocket.send(msgpack_numpy.packb(message))
            response_raw = await websocket.recv()
            if isinstance(response_raw, str):
                raise InferenceTransportError("deployment_inference_failed")
            return json_safe(msgpack_numpy.unpackb(response_raw)), metadata_value

    try:
        response, metadata = await asyncio.wait_for(exchange(), timeout=timeout)
    except asyncio.TimeoutError as exc:
        raise InferenceTransportError("deployment_inference_timeout") from exc
    except InferenceTransportError:
        raise
    except Exception as exc:
        # The caller receives a stable code; arbitrary remote/transport text is
        # not reflected because it can contain model internals or credentials.
        raise InferenceTransportError("deployment_connection_failed") from exc
    if not isinstance(response, dict):
        raise InferenceTransportError("invalid_deployment_response")
    if response.get("request_id") not in {None, request.request_id}:
        raise InferenceTransportError("deployment_request_id_mismatch")
    if response.get("ok") is False or response.get("status") == "error":
        raise InferenceTransportError("deployment_inference_failed")
    serialized = json.dumps(response, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode("utf-8")
    if len(serialized) > MAX_OUTPUT_BYTES:
        raise InferenceTransportError("inference_output_too_large")
    return response, metadata if isinstance(metadata, dict) else {}, (time.perf_counter() - started) * 1000


class InferenceCoordinator:
    """Enforce one active Playground request per managed deployment."""

    def __init__(self) -> None:
        self._locks: dict[str, asyncio.Lock] = {}

    async def run(self, deployment_id: str, operation):  # type: ignore[no-untyped-def]
        lock = self._locks.setdefault(deployment_id, asyncio.Lock())
        if lock.locked():
            raise InferenceInputError("deployment_inference_busy", status_code=409)
        await lock.acquire()
        try:
            return await operation()
        finally:
            lock.release()


def save_inference_inputs(state_dir: Path, run_id: str, images: Sequence[UploadedImage]) -> list[str]:
    if not images:
        return []
    root = state_dir / "inference" / run_id / "inputs"
    root.mkdir(mode=0o700, parents=True, exist_ok=False)
    result: list[str] = []
    extensions = {"image/jpeg": ".jpg", "image/png": ".png", "image/webp": ".webp"}
    for index, image in enumerate(images):
        path = root / f"image-{index + 1:02d}{extensions[image.content_type]}"
        path.write_bytes(image.data)
        path.chmod(0o600)
        result.append(str(path))
    return result
