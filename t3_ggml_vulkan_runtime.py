from __future__ import annotations

import ctypes
import os
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from transformers import (
    LogitsProcessorList,
    RepetitionPenaltyLogitsProcessor,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from transformers.cache_utils import DynamicCache


ROOT = Path(__file__).resolve().parent
DEFAULT_WEIGHTS_DIR = ROOT / "exports" / "ggml_t3_real_prompt_multistep_chunk270_s4_p935"
DEFAULT_LIB_PATH = ROOT / "libt3_ggml_vulkan_bridge.so"
DEFAULT_NATIVE_SAMPLER_LIB_PATH = ROOT / "libt3_native_sampler_bridge.so"
DEFAULT_GGML_LIB_DIR = Path("/root/llama.cpp/build-vulkan/bin")
UINT64_MASK = (1 << 64) - 1


def _legacy_past(past_key_values: Any):
    if hasattr(past_key_values, "to_legacy_cache"):
        return past_key_values.to_legacy_cache()
    return past_key_values


@contextmanager
def _temporary_torch_threads(num_threads: int | None):
    if num_threads is None or num_threads <= 0:
        yield
        return
    previous = torch.get_num_threads()
    if previous != num_threads:
        torch.set_num_threads(num_threads)
    try:
        yield
    finally:
        if previous != num_threads:
            torch.set_num_threads(previous)


class T3GGMLVulkanRuntime:
    """ctypes wrapper for the ggml/Vulkan Chatterbox T3 one-token stepper."""

    def __init__(
        self,
        weights_dir: str | os.PathLike[str] = DEFAULT_WEIGHTS_DIR,
        lib_path: str | os.PathLike[str] = DEFAULT_LIB_PATH,
        ggml_lib_dir: str | os.PathLike[str] = DEFAULT_GGML_LIB_DIR,
        device_index: int = 0,
    ) -> None:
        self.weights_dir = Path(weights_dir)
        self.lib_path = Path(lib_path)
        self.ggml_lib_dir = Path(ggml_lib_dir)
        if not self.weights_dir.exists():
            raise FileNotFoundError(f"ggml T3 weights directory not found: {self.weights_dir}")
        if not self.lib_path.exists():
            raise FileNotFoundError(f"ggml T3 bridge library not found: {self.lib_path}")

        self.lib = ctypes.CDLL(str(self.lib_path))
        self._bind()

        err = ctypes.create_string_buffer(4096)
        self.handle = self.lib.cb_t3_ggml_create(
            str(self.weights_dir).encode(),
            str(self.ggml_lib_dir).encode(),
            int(device_index),
            err,
            ctypes.sizeof(err),
        )
        if not self.handle:
            raise RuntimeError(err.value.decode("utf-8", errors="replace"))

        self.hidden = int(self.lib.cb_t3_ggml_hidden())
        self.layers = int(self.lib.cb_t3_ggml_layers())
        self.heads = int(self.lib.cb_t3_ggml_heads())
        self.head_dim = int(self.lib.cb_t3_ggml_head_dim())
        self.max_len = int(self.lib.cb_t3_ggml_max_len())
        self.seq_len = int(self.lib.cb_t3_ggml_seq_len())
        self.speech_vocab = int(self.lib.cb_t3_ggml_speech_vocab())
        self.device = self.lib.cb_t3_ggml_device(self.handle).decode("utf-8", errors="replace")

    def _bind(self) -> None:
        c_char_p = ctypes.c_char_p
        c_int = ctypes.c_int
        c_size_t = ctypes.c_size_t
        c_void_p = ctypes.c_void_p
        c_double_p = ctypes.POINTER(ctypes.c_double)
        c_float_p = ctypes.POINTER(ctypes.c_float)

        self.lib.cb_t3_ggml_create.argtypes = [c_char_p, c_char_p, c_int, c_char_p, c_size_t]
        self.lib.cb_t3_ggml_create.restype = c_void_p
        self.lib.cb_t3_ggml_destroy.argtypes = [c_void_p]
        self.lib.cb_t3_ggml_destroy.restype = None
        self.lib.cb_t3_ggml_device.argtypes = [c_void_p]
        self.lib.cb_t3_ggml_device.restype = c_char_p
        self.lib.cb_t3_ggml_hidden.restype = c_int
        self.lib.cb_t3_ggml_layers.restype = c_int
        self.lib.cb_t3_ggml_heads.restype = c_int
        self.lib.cb_t3_ggml_head_dim.restype = c_int
        self.lib.cb_t3_ggml_max_len.restype = c_int
        self.lib.cb_t3_ggml_seq_len.restype = c_int
        self.lib.cb_t3_ggml_speech_vocab.restype = c_int
        self.lib.cb_t3_ggml_set_layer_cache.argtypes = [
            c_void_p,
            c_int,
            c_float_p,
            c_float_p,
            c_char_p,
            c_size_t,
        ]
        self.lib.cb_t3_ggml_set_layer_cache.restype = c_int
        self._set_layer_cache_prefix = None
        try:
            set_layer_cache_prefix = self.lib.cb_t3_ggml_set_layer_cache_prefix
        except AttributeError:
            pass
        else:
            set_layer_cache_prefix.argtypes = [
                c_void_p,
                c_int,
                c_float_p,
                c_float_p,
                c_int,
                c_char_p,
                c_size_t,
            ]
            set_layer_cache_prefix.restype = c_int
            self._set_layer_cache_prefix = set_layer_cache_prefix
        self._set_layer_cache_range = None
        try:
            set_layer_cache_range = self.lib.cb_t3_ggml_set_layer_cache_range
        except AttributeError:
            pass
        else:
            set_layer_cache_range.argtypes = [
                c_void_p,
                c_int,
                c_float_p,
                c_float_p,
                c_int,
                c_int,
                c_char_p,
                c_size_t,
            ]
            set_layer_cache_range.restype = c_int
            self._set_layer_cache_range = set_layer_cache_range
        self.lib.cb_t3_ggml_run_step.argtypes = [
            c_void_p,
            c_float_p,
            c_float_p,
            c_int,
            c_float_p,
            c_double_p,
            c_char_p,
            c_size_t,
        ]
        self.lib.cb_t3_ggml_run_step.restype = c_int

    def close(self) -> None:
        handle = getattr(self, "handle", None)
        if handle:
            self.lib.cb_t3_ggml_destroy(handle)
            self.handle = None

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass

    def set_layer_cache_arrays(self, layer: int, key_cache: np.ndarray, value_cache: np.ndarray) -> None:
        key = np.ascontiguousarray(key_cache, dtype=np.float32).reshape(-1)
        value = np.ascontiguousarray(value_cache, dtype=np.float32).reshape(-1)
        expected = self.heads * self.max_len * self.head_dim
        if key.size != expected or value.size != expected:
            raise ValueError(f"cache arrays must contain {expected} float32 values")
        err = ctypes.create_string_buffer(4096)
        rc = self.lib.cb_t3_ggml_set_layer_cache(
            self.handle,
            int(layer),
            key.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            err,
            ctypes.sizeof(err),
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", errors="replace"))

    def set_layer_cache_prefix_arrays(
        self,
        layer: int,
        key_prefix: np.ndarray,
        value_prefix: np.ndarray,
        valid_len: int,
    ) -> None:
        if self._set_layer_cache_prefix is None:
            raise RuntimeError("ggml T3 bridge does not expose prefix cache upload")
        if valid_len > self.max_len:
            raise ValueError(f"valid_len {valid_len} exceeds ggml cache max_len {self.max_len}")
        key = np.ascontiguousarray(key_prefix, dtype=np.float32).reshape(-1)
        value = np.ascontiguousarray(value_prefix, dtype=np.float32).reshape(-1)
        expected = self.heads * valid_len * self.head_dim
        if key.size != expected or value.size != expected:
            raise ValueError(f"cache prefix arrays must contain {expected} float32 values")
        err = ctypes.create_string_buffer(4096)
        rc = self._set_layer_cache_prefix(
            self.handle,
            int(layer),
            key.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(valid_len),
            err,
            ctypes.sizeof(err),
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", errors="replace"))

    def set_layer_cache_range_arrays(
        self,
        layer: int,
        key_segment: np.ndarray,
        value_segment: np.ndarray,
        start_pos: int,
        segment_len: int,
    ) -> None:
        if self._set_layer_cache_range is None:
            raise RuntimeError("ggml T3 bridge does not expose range cache upload")
        if start_pos < 0 or segment_len < 0 or start_pos + segment_len > self.max_len:
            raise ValueError(
                f"cache range {start_pos}..{start_pos + segment_len} exceeds ggml cache max_len {self.max_len}"
            )
        key = np.ascontiguousarray(key_segment, dtype=np.float32).reshape(-1)
        value = np.ascontiguousarray(value_segment, dtype=np.float32).reshape(-1)
        expected = self.heads * segment_len * self.head_dim
        if key.size != expected or value.size != expected:
            raise ValueError(f"cache range arrays must contain {expected} float32 values")
        err = ctypes.create_string_buffer(4096)
        rc = self._set_layer_cache_range(
            self.handle,
            int(layer),
            key.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            value.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(start_pos),
            int(segment_len),
            err,
            ctypes.sizeof(err),
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", errors="replace"))

    def set_cache_from_past(self, past_key_values: Any, valid_len: int) -> None:
        if valid_len > self.max_len:
            raise ValueError(f"valid_len {valid_len} exceeds ggml cache max_len {self.max_len}")
        for layer, layer_cache in enumerate(_legacy_past(past_key_values)):
            key, value = layer_cache[0], layer_cache[1]
            key_cpu = key[0].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            value_cpu = value[0].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
            if key_cpu.shape != (self.heads, valid_len, self.head_dim):
                raise ValueError(f"unexpected key shape for layer {layer}: {key_cpu.shape}")
            if value_cpu.shape != (self.heads, valid_len, self.head_dim):
                raise ValueError(f"unexpected value shape for layer {layer}: {value_cpu.shape}")

            if self._set_layer_cache_prefix is not None:
                self.set_layer_cache_prefix_arrays(layer, key_cpu, value_cpu, valid_len)
            else:
                key_cache = np.zeros((self.heads, self.max_len, self.head_dim), dtype=np.float32)
                value_cache = np.zeros((self.heads, self.max_len, self.head_dim), dtype=np.float32)
                key_cache[:, :valid_len, :] = key_cpu
                value_cache[:, :valid_len, :] = value_cpu
                self.set_layer_cache_arrays(layer, key_cache, value_cache)

    def set_cache_range_from_past(self, past_key_values: Any, start_pos: int, segment_len: int) -> None:
        if segment_len == 0:
            return
        if self._set_layer_cache_range is None:
            self.set_cache_from_past(past_key_values, start_pos + segment_len)
            return
        if start_pos < 0 or segment_len < 0 or start_pos + segment_len > self.max_len:
            raise ValueError(
                f"cache range {start_pos}..{start_pos + segment_len} exceeds ggml cache max_len {self.max_len}"
            )
        for layer, layer_cache in enumerate(_legacy_past(past_key_values)):
            key, value = layer_cache[0], layer_cache[1]
            key_cpu = (
                key[0, :, start_pos : start_pos + segment_len, :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            value_cpu = (
                value[0, :, start_pos : start_pos + segment_len, :]
                .detach()
                .to(device="cpu", dtype=torch.float32)
                .contiguous()
                .numpy()
            )
            if key_cpu.shape != (self.heads, segment_len, self.head_dim):
                raise ValueError(f"unexpected key range shape for layer {layer}: {key_cpu.shape}")
            if value_cpu.shape != (self.heads, segment_len, self.head_dim):
                raise ValueError(f"unexpected value range shape for layer {layer}: {value_cpu.shape}")
            self.set_layer_cache_range_arrays(layer, key_cpu, value_cpu, start_pos, segment_len)

    def mask_for_valid_len(self, valid_len: int) -> np.ndarray:
        if valid_len < 0 or valid_len >= self.max_len:
            raise ValueError(f"valid_len {valid_len} is outside cache range 0..{self.max_len - 1}")
        mask = np.full((self.seq_len,), -1.0e9, dtype=np.float32)
        mask[:valid_len] = 0.0
        mask[self.max_len] = 0.0
        return mask

    def masks_for_valid_lens(self, first_valid_len: int, count: int) -> np.ndarray:
        if count < 0:
            raise ValueError("count must be non-negative")
        if count == 0:
            return np.empty((0, self.seq_len), dtype=np.float32)
        last_valid_len = first_valid_len + count - 1
        if first_valid_len < 0 or last_valid_len >= self.max_len:
            raise ValueError(
                f"valid_len range {first_valid_len}..{last_valid_len} is outside "
                f"cache range 0..{self.max_len - 1}"
            )
        masks = np.full((count, self.seq_len), -1.0e9, dtype=np.float32)
        masks[:, self.max_len] = 0.0
        for index, valid_len in enumerate(range(first_valid_len, first_valid_len + count)):
            masks[index, :valid_len] = 0.0
        return masks

    def run_step_with_mask(
        self,
        input_hidden: np.ndarray,
        attn_mask: np.ndarray,
        slot_index: int,
    ) -> tuple[np.ndarray, float]:
        inp = np.ascontiguousarray(input_hidden, dtype=np.float32).reshape(-1)
        mask = np.ascontiguousarray(attn_mask, dtype=np.float32).reshape(-1)
        if inp.size != self.hidden:
            raise ValueError(f"input_hidden must contain {self.hidden} values")
        if mask.size != self.seq_len:
            raise ValueError(f"attn_mask must contain {self.seq_len} values")

        logits = np.empty((self.speech_vocab,), dtype=np.float32)
        elapsed_ms = ctypes.c_double(0.0)
        err = ctypes.create_string_buffer(4096)
        rc = self.lib.cb_t3_ggml_run_step(
            self.handle,
            inp.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            mask.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(slot_index),
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.byref(elapsed_ms),
            err,
            ctypes.sizeof(err),
        )
        if rc != 0:
            raise RuntimeError(err.value.decode("utf-8", errors="replace"))
        return logits, float(elapsed_ms.value)

    def run_step(self, input_hidden: np.ndarray, valid_len: int) -> tuple[np.ndarray, float]:
        return self.run_step_with_mask(input_hidden, self.mask_for_valid_len(valid_len), valid_len)


def _logits_processors(temperature: float, top_k: int, top_p: float, repetition_penalty: float) -> LogitsProcessorList:
    processors = LogitsProcessorList()
    if temperature > 0 and temperature != 1.0:
        processors.append(TemperatureLogitsWarper(temperature))
    if top_k > 0:
        processors.append(TopKLogitsWarper(top_k))
    if top_p < 1.0:
        processors.append(TopPLogitsWarper(top_p))
    if repetition_penalty != 1.0:
        processors.append(RepetitionPenaltyLogitsProcessor(repetition_penalty))
    return processors


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


class T3NativeSampler:
    """ctypes wrapper for the opt-in native T3 sampler bridge."""

    def __init__(self, lib_path: str | os.PathLike[str] = DEFAULT_NATIVE_SAMPLER_LIB_PATH) -> None:
        self.lib_path = Path(lib_path)
        if not self.lib_path.exists():
            raise FileNotFoundError(f"T3 native sampler bridge library not found: {self.lib_path}")
        self.lib = ctypes.CDLL(str(self.lib_path))
        self._bind()

    def _bind(self) -> None:
        self.lib.cb_t3_native_sample.argtypes = [
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int,
            ctypes.c_uint64,
            ctypes.c_float,
            ctypes.c_int,
            ctypes.c_float,
            ctypes.c_float,
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.cb_t3_native_sample.restype = ctypes.c_int

    def sample(
        self,
        logits: np.ndarray,
        input_ids: np.ndarray,
        *,
        seed: int,
        temperature: float,
        top_k: int,
        top_p: float,
        repetition_penalty: float,
    ) -> int | None:
        logits = np.ascontiguousarray(logits, dtype=np.float32).reshape(-1)
        input_ids = np.ascontiguousarray(input_ids, dtype=np.int32).reshape(-1)
        effective_top_k = int(top_k) if int(top_k) > 0 else int(logits.size)
        out_token = ctypes.c_int32(-1)
        rc = self.lib.cb_t3_native_sample(
            logits.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            int(logits.size),
            input_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            int(input_ids.size),
            ctypes.c_uint64(seed & UINT64_MASK),
            ctypes.c_float(float(temperature)),
            effective_top_k,
            ctypes.c_float(float(top_p)),
            ctypes.c_float(float(repetition_penalty)),
            ctypes.byref(out_token),
        )
        if rc == 1:
            return None
        if rc != 0:
            raise RuntimeError(f"T3 native sampler failed with rc={rc}")
        return int(out_token.value)


def _get_native_sampler(runtime: T3GGMLVulkanRuntime) -> T3NativeSampler:
    sampler = getattr(runtime, "_t3_native_sampler", None)
    if sampler is None:
        lib_path = os.getenv("CHATTERBOX_T3_NATIVE_SAMPLER_LIB_PATH", str(DEFAULT_NATIVE_SAMPLER_LIB_PATH))
        sampler = T3NativeSampler(lib_path)
        runtime._t3_native_sampler = sampler
    return sampler


def _next_native_sampler_seed(runtime: T3GGMLVulkanRuntime) -> int:
    state = getattr(runtime, "_t3_native_sampler_seed_state", None)
    if state is None:
        env_seed = os.getenv("CHATTERBOX_T3_NATIVE_SAMPLER_SEED")
        state = int(env_seed, 0) if env_seed else time.time_ns()
    state = (int(state) + 0x9E3779B97F4A7C15) & UINT64_MASK
    runtime._t3_native_sampler_seed_state = state
    z = state
    z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & UINT64_MASK
    z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & UINT64_MASK
    return (z ^ (z >> 31)) & UINT64_MASK


def _sample_with_processors(
    processors: LogitsProcessorList,
    input_ids: torch.LongTensor,
    speech_logits: torch.Tensor,
) -> torch.LongTensor | None:
    processed_logits = processors(input_ids, speech_logits)
    if torch.all(processed_logits == -float("inf")):
        return None
    probs = F.softmax(processed_logits, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _sample_fast(
    input_ids: torch.LongTensor,
    speech_logits: torch.Tensor,
    *,
    temperature: float,
    top_k: int,
    top_p: float,
    repetition_penalty: float,
) -> torch.LongTensor | None:
    scores = speech_logits
    if temperature > 0 and temperature != 1.0:
        scores = scores / temperature
    if top_k > 0:
        top_k = min(top_k, scores.size(-1))
        top_values, top_indices = torch.topk(scores, top_k)
        top_k_threshold = top_values[..., -1, None]
        indices_to_remove = scores < top_k_threshold
        if top_p < 1.0 and scores.size(0) == 1:
            sorted_logits, sorted_order = torch.sort(top_values, descending=False)
            cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
            sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
            sorted_indices_to_remove[..., -1:] = 0
            top_indices_to_remove = torch.zeros_like(sorted_indices_to_remove)
            top_indices_to_remove = top_indices_to_remove.scatter(
                1,
                sorted_order,
                sorted_indices_to_remove,
            )
            scores = torch.full_like(scores, -float("inf"))
            scores = scores.scatter(1, top_indices, top_values.masked_fill(top_indices_to_remove, -float("inf")))
        else:
            scores = scores.masked_fill(indices_to_remove, -float("inf"))
    if top_p < 1.0 and not (top_k > 0 and scores.size(0) == 1):
        sorted_logits, sorted_indices = torch.sort(scores, descending=False)
        cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)
        sorted_indices_to_remove = cumulative_probs <= (1 - top_p)
        sorted_indices_to_remove[..., -1:] = 0
        indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
        scores = scores.masked_fill(indices_to_remove, -float("inf"))
    if repetition_penalty != 1.0:
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(1)
        score = torch.gather(scores, 1, input_ids)
        score = torch.where(score < 0, score * repetition_penalty, score / repetition_penalty)
        scores = scores.scatter(1, input_ids, score)
    if torch.all(scores == -float("inf")):
        return None
    probs = F.softmax(scores, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def _prefix_cache_key(t3, t3_cond: Any, len_cond: int, prefill_threads: int | None, embeds: torch.Tensor) -> tuple:
    return (
        id(t3),
        id(t3_cond),
        int(len_cond),
        int(prefill_threads or 0),
        str(embeds.device),
        str(embeds.dtype),
    )


def _clone_reusable_past(past_key_values: Any) -> tuple:
    cloned_layers = []
    for layer in _legacy_past(past_key_values):
        cloned_layers.append(
            tuple(item.detach().clone() if torch.is_tensor(item) else item for item in layer)
        )
    return tuple(cloned_layers)


def _dynamic_cache_from_reusable_past(past_key_values: Any) -> DynamicCache:
    return DynamicCache(ddp_cache_data=_clone_reusable_past(past_key_values))


def _prefill_with_optional_prefix_cache(
    t3,
    runtime: T3GGMLVulkanRuntime,
    embeds: torch.Tensor,
    len_cond: int,
    t3_cond: Any,
    prefill_threads: int | None,
    use_prefix_prefill: bool,
):
    if not use_prefix_prefill or len_cond <= 0:
        perf_started = time.perf_counter()
        with _temporary_torch_threads(prefill_threads):
            outputs = t3.tfmr(inputs_embeds=embeds, use_cache=True)
        return outputs, {
            "prefix_prefill": False,
            "prefix_cache_hit": False,
            "len_cond": int(len_cond),
            "suffix_len": int(embeds.shape[1]),
            "full_prefill_seconds": time.perf_counter() - perf_started,
            "prefix_build_seconds": 0.0,
            "suffix_prefill_seconds": 0.0,
        }

    cache_key = _prefix_cache_key(t3, t3_cond, len_cond, prefill_threads, embeds)
    cache = getattr(runtime, "_t3_prefix_prefill_cache", None)
    cache_hit = bool(cache is not None and cache.get("key") == cache_key)
    prefix_build_seconds = 0.0
    if not cache_hit:
        cond_embeds = embeds[:, :len_cond].contiguous()
        perf_started = time.perf_counter()
        with _temporary_torch_threads(prefill_threads):
            cond_outputs = t3.tfmr(inputs_embeds=cond_embeds, use_cache=True)
        prefix_build_seconds = time.perf_counter() - perf_started
        cache = {
            "key": cache_key,
            "past_key_values": _clone_reusable_past(cond_outputs.past_key_values),
            "len_cond": int(len_cond),
        }
        runtime._t3_prefix_prefill_cache = cache

    suffix_embeds = embeds[:, len_cond:].contiguous()
    prefix_past = _dynamic_cache_from_reusable_past(cache["past_key_values"])
    perf_started = time.perf_counter()
    with _temporary_torch_threads(prefill_threads):
        outputs = t3.tfmr(
            inputs_embeds=suffix_embeds,
            past_key_values=prefix_past,
            use_cache=True,
        )
    suffix_prefill_seconds = time.perf_counter() - perf_started
    return outputs, {
        "prefix_prefill": True,
        "prefix_cache_hit": cache_hit,
        "len_cond": int(len_cond),
        "suffix_len": int(suffix_embeds.shape[1]),
        "full_prefill_seconds": 0.0,
        "prefix_build_seconds": prefix_build_seconds,
        "suffix_prefill_seconds": suffix_prefill_seconds,
    }


@torch.inference_mode()
def warm_prefix_prefill_cache(
    t3,
    runtime: T3GGMLVulkanRuntime,
    *,
    t3_cond,
    text_tokens: torch.LongTensor,
    prefill_threads: int | None = None,
) -> dict[str, Any]:
    """Build and upload the reusable conditioning prefix cache without generating audio."""

    if text_tokens.size(0) != 1:
        raise ValueError("ggml Vulkan T3 runtime only supports batch size 1")

    text_tokens = text_tokens.to(t3.device)
    if prefill_threads is None:
        prefill_threads = int(os.getenv("CHATTERBOX_T3_PREFILL_THREADS", "0") or "0")
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    started = time.perf_counter()
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )
    prepare_seconds = time.perf_counter() - started
    if len_cond <= 0:
        raise ValueError("T3 conditioning prefix is empty")

    cache_key = _prefix_cache_key(t3, t3_cond, len_cond, prefill_threads, embeds)
    cache = getattr(runtime, "_t3_prefix_prefill_cache", None)
    prefix_cache_hit = bool(cache is not None and cache.get("key") == cache_key)
    prefix_build_seconds = 0.0
    if not prefix_cache_hit:
        cond_embeds = embeds[:, :len_cond].contiguous()
        started = time.perf_counter()
        with _temporary_torch_threads(prefill_threads):
            cond_outputs = t3.tfmr(inputs_embeds=cond_embeds, use_cache=True)
        prefix_build_seconds = time.perf_counter() - started
        cache = {
            "key": cache_key,
            "past_key_values": _clone_reusable_past(cond_outputs.past_key_values),
            "len_cond": int(len_cond),
        }
        runtime._t3_prefix_prefill_cache = cache

    ggml_prefix_was_resident = getattr(runtime, "_t3_ggml_prefix_cache_resident_key", None) == cache_key
    upload_seconds = 0.0
    if not ggml_prefix_was_resident:
        started = time.perf_counter()
        runtime.set_cache_from_past(cache["past_key_values"], int(len_cond))
        upload_seconds = time.perf_counter() - started
        runtime._t3_ggml_prefix_cache_resident_key = cache_key

    info = {
        "prefix_cache_hit": prefix_cache_hit,
        "ggml_prefix_was_resident": ggml_prefix_was_resident,
        "len_cond": int(len_cond),
        "prepare_embeds_seconds": prepare_seconds,
        "prefix_build_seconds": prefix_build_seconds,
        "ggml_prefix_upload_seconds": upload_seconds,
        "prefill_threads": int(prefill_threads or 0),
        "cache_range_upload_available": runtime._set_layer_cache_range is not None,
    }
    runtime.last_prefix_warmup_info = dict(info)
    return info


@torch.inference_mode()
def inference_turbo_vulkan(
    t3,
    runtime: T3GGMLVulkanRuntime,
    *,
    t3_cond,
    text_tokens: torch.LongTensor,
    temperature: float = 0.8,
    top_k: int = 1000,
    top_p: float = 0.95,
    repetition_penalty: float = 1.2,
    max_gen_len: int = 1000,
    fast_loop: bool | None = None,
    prefill_threads: int | None = None,
    fast_sampler: bool | None = None,
    native_sampler: bool | None = None,
    native_sampler_seed: int | None = None,
    prefix_prefill: bool | None = None,
) -> torch.LongTensor:
    """Drop-in experimental replacement for T3.inference_turbo's repeated token loop.

    PyTorch still performs conditioning and the initial prefill. The repeated
    one-token GPT-2 block stack runs through ggml/Vulkan and keeps its KV cache
    resident in the ggml backend.
    """

    if text_tokens.size(0) != 1:
        raise ValueError("ggml Vulkan T3 runtime only supports batch size 1")

    runtime.last_prefill_info = None
    runtime.last_loop_info = None
    text_tokens = text_tokens.to(t3.device)
    processors = _logits_processors(temperature, top_k, top_p, repetition_penalty)
    if fast_sampler is None:
        fast_sampler = _env_flag("CHATTERBOX_T3_FAST_SAMPLER")
    if native_sampler is None:
        native_sampler = _env_flag("CHATTERBOX_T3_NATIVE_SAMPLER")
    native_sampler_bridge = _get_native_sampler(runtime) if native_sampler else None
    if native_sampler_bridge is not None:
        if native_sampler_seed is None:
            env_seed = os.getenv("CHATTERBOX_T3_NATIVE_SAMPLER_SEED")
            native_sampler_seed = int(env_seed, 0) if env_seed else int(torch.initial_seed())
        runtime._t3_native_sampler_seed_state = int(native_sampler_seed) & UINT64_MASK
    speech_start_token = t3.hp.start_speech_token * torch.ones_like(text_tokens[:, :1])
    embeds, len_cond = t3.prepare_input_embeds(
        t3_cond=t3_cond,
        text_tokens=text_tokens,
        speech_tokens=speech_start_token,
        cfg_weight=0.0,
    )

    if prefill_threads is None:
        prefill_threads = int(os.getenv("CHATTERBOX_T3_PREFILL_THREADS", "0") or "0")
    if prefix_prefill is None:
        prefix_prefill = _env_flag("CHATTERBOX_T3_PREFIX_PREFILL")
    llm_outputs, _prefill_info = _prefill_with_optional_prefix_cache(
        t3,
        runtime,
        embeds,
        len_cond,
        t3_cond,
        prefill_threads,
        prefix_prefill,
    )
    runtime.last_prefill_info = dict(_prefill_info)
    hidden_states = llm_outputs[0]
    speech_logits = t3.speech_head(hidden_states[:, -1:])
    past_key_values = llm_outputs.past_key_values
    initial_context_len = int(embeds.shape[1])
    if initial_context_len >= runtime.max_len:
        raise ValueError(
            f"initial T3 context {initial_context_len} exceeds ggml Vulkan cache max_len {runtime.max_len}"
        )

    if fast_sampler:
        next_speech_token = _sample_fast(
            speech_start_token,
            speech_logits[:, -1, :],
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            repetition_penalty=repetition_penalty,
        )
    else:
        next_speech_token = _sample_with_processors(processors, speech_start_token, speech_logits[:, -1, :])
    if next_speech_token is None:
        return torch.empty((1, 0), dtype=text_tokens.dtype, device=text_tokens.device)

    cache_upload_started = time.perf_counter()
    cache_upload_mode = "full"
    cache_upload_start_pos = 0
    cache_upload_len = initial_context_len
    prefix_cache = getattr(runtime, "_t3_prefix_prefill_cache", None)
    prefix_cache_key = prefix_cache.get("key") if prefix_cache is not None else None
    ggml_prefix_resident = (
        bool(prefix_prefill)
        and prefix_cache_key is not None
        and getattr(runtime, "_t3_ggml_prefix_cache_resident_key", None) == prefix_cache_key
    )
    if (
        bool(prefix_prefill)
        and bool(_prefill_info.get("prefix_cache_hit"))
        and ggml_prefix_resident
        and runtime._set_layer_cache_range is not None
    ):
        suffix_start = int(_prefill_info["len_cond"])
        suffix_len = initial_context_len - suffix_start
        runtime.set_cache_range_from_past(past_key_values, suffix_start, suffix_len)
        cache_upload_mode = "suffix_range"
        cache_upload_start_pos = suffix_start
        cache_upload_len = suffix_len
    else:
        runtime.set_cache_from_past(past_key_values, initial_context_len)
        if bool(prefix_prefill) and prefix_cache_key is not None:
            runtime._t3_ggml_prefix_cache_resident_key = prefix_cache_key
    runtime.last_prefill_info.update(
        {
            "ggml_prefix_resident_before_upload": ggml_prefix_resident,
            "cache_upload_mode": cache_upload_mode,
            "cache_upload_start_pos": int(cache_upload_start_pos),
            "cache_upload_len": int(cache_upload_len),
            "cache_upload_seconds": time.perf_counter() - cache_upload_started,
            "cache_range_upload_available": runtime._set_layer_cache_range is not None,
        }
    )

    generated_speech_tokens = torch.empty(
        next_speech_token.size(0),
        max_gen_len + 1,
        dtype=next_speech_token.dtype,
        device=next_speech_token.device,
    )
    generated_speech_tokens[:, 0:1] = next_speech_token
    generated_count = 1
    current_speech_token = next_speech_token
    current_token_id = int(next_speech_token.detach().cpu().reshape(-1)[0].item())
    native_input_ids = np.empty((max_gen_len + 1,), dtype=np.int32) if native_sampler_bridge is not None else None
    if native_input_ids is not None:
        native_input_ids[0] = current_token_id
    native_fast_token_buffer = bool(native_sampler_bridge is not None and fast_loop)
    loop_timings = {
        "fast_loop_setup_seconds": 0.0,
        "step_hidden_seconds": 0.0,
        "step_ggml_wall_seconds": 0.0,
        "step_ggml_reported_seconds": 0.0,
        "step_logits_to_torch_seconds": 0.0,
        "step_sampling_seconds": 0.0,
        "step_native_sampler_seconds": 0.0,
    }
    ggml_wall_times: list[float] = []
    ggml_reported_ms: list[float] = []
    if fast_loop is None:
        fast_loop = os.getenv("CHATTERBOX_T3_FAST_LOOP", "").lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    max_steps = min(max_gen_len, runtime.max_len - initial_context_len)
    precomputed_masks = runtime.masks_for_valid_lens(initial_context_len, max_steps) if fast_loop else None
    speech_embedding_np = None
    position_embedding_np = None
    if fast_loop:
        setup_start = time.perf_counter()
        if not hasattr(t3.tfmr, "wpe"):
            raise ValueError("fast ggml T3 loop currently requires GPT-2 position embeddings")
        speech_embedding_np = t3.speech_emb.weight.detach().to(device="cpu", dtype=torch.float32).numpy()
        position_embedding_np = t3.tfmr.wpe.weight.detach().to(device="cpu", dtype=torch.float32).numpy()
        loop_timings["fast_loop_setup_seconds"] = time.perf_counter() - setup_start

    for _ in range(max_gen_len):
        slot_index = initial_context_len + generated_count - 1
        if slot_index >= runtime.max_len:
            break

        hidden_start = time.perf_counter()
        if fast_loop:
            assert precomputed_masks is not None
            assert speech_embedding_np is not None
            assert position_embedding_np is not None
            current_hidden_np = speech_embedding_np[current_token_id] + position_embedding_np[slot_index]
            attn_mask = precomputed_masks[generated_count - 1]
        else:
            current_speech_embed = t3.speech_emb(current_speech_token)
            position_ids = torch.full(
                (current_speech_embed.shape[0], current_speech_embed.shape[1]),
                slot_index,
                dtype=torch.long,
                device=current_speech_embed.device,
            )
            current_hidden = current_speech_embed + t3.tfmr.wpe(position_ids)
            current_hidden_np = current_hidden[0, 0].detach().to(device="cpu", dtype=torch.float32).numpy()
            attn_mask = None
        loop_timings["step_hidden_seconds"] += time.perf_counter() - hidden_start

        ggml_start = time.perf_counter()
        if attn_mask is None:
            logits_np, _elapsed_ms = runtime.run_step(current_hidden_np, slot_index)
        else:
            logits_np, _elapsed_ms = runtime.run_step_with_mask(current_hidden_np, attn_mask, slot_index)
        ggml_wall = time.perf_counter() - ggml_start
        ggml_wall_times.append(ggml_wall)
        ggml_reported_ms.append(_elapsed_ms)
        loop_timings["step_ggml_wall_seconds"] += ggml_wall
        loop_timings["step_ggml_reported_seconds"] += _elapsed_ms / 1000.0

        sample_start = time.perf_counter()
        next_token_id: int | None = None
        if native_sampler_bridge is not None:
            assert native_input_ids is not None
            next_token_id = native_sampler_bridge.sample(
                logits_np,
                native_input_ids[:generated_count],
                seed=_next_native_sampler_seed(runtime),
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty,
            )
            if next_token_id is None:
                next_speech_token = None
            elif native_fast_token_buffer:
                next_speech_token = None
            else:
                next_speech_token = torch.tensor(
                    [[next_token_id]],
                    dtype=generated_speech_tokens.dtype,
                    device=generated_speech_tokens.device,
                )
            loop_timings["step_native_sampler_seconds"] += time.perf_counter() - sample_start
        else:
            logits_start = time.perf_counter()
            speech_logits = torch.from_numpy(logits_np).to(device=generated_speech_tokens.device).unsqueeze(0)
            loop_timings["step_logits_to_torch_seconds"] += time.perf_counter() - logits_start
            input_ids = generated_speech_tokens[:, :generated_count]
            if fast_sampler:
                next_speech_token = _sample_fast(
                    input_ids,
                    speech_logits,
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty,
                )
            else:
                next_speech_token = _sample_with_processors(processors, input_ids, speech_logits)
            if next_speech_token is not None:
                next_token_id = int(next_speech_token.detach().cpu().reshape(-1)[0].item())
        loop_timings["step_sampling_seconds"] += time.perf_counter() - sample_start
        if next_token_id is None or (next_speech_token is None and not native_fast_token_buffer):
            break
        if native_fast_token_buffer:
            generated_speech_tokens[:, generated_count] = next_token_id
        else:
            generated_speech_tokens[:, generated_count : generated_count + 1] = next_speech_token
        if native_input_ids is not None:
            native_input_ids[generated_count] = next_token_id
        generated_count += 1
        if not native_fast_token_buffer:
            current_speech_token = next_speech_token
        current_token_id = next_token_id
        if next_token_id == t3.hp.stop_speech_token:
            break

    all_tokens = generated_speech_tokens[:, :generated_count]
    if all_tokens.size(1) > 0 and all_tokens[0, -1] == t3.hp.stop_speech_token:
        all_tokens = all_tokens[:, :-1]
    loop_iterations = len(ggml_wall_times)
    runtime.last_loop_info = {
        "fast_loop": bool(fast_loop),
        "fast_sampler": bool(fast_sampler),
        "native_sampler": bool(native_sampler_bridge is not None),
        "native_fast_token_buffer": bool(native_fast_token_buffer),
        "native_sampler_lib": (
            native_sampler_bridge.lib_path.as_posix() if native_sampler_bridge is not None else None
        ),
        "native_sampler_seed": int(native_sampler_seed) if native_sampler_bridge is not None else None,
        "compact_sampler": False,
        "initial_context_len": int(initial_context_len),
        "max_steps": int(max_steps),
        "loop_iterations": int(loop_iterations),
        "generated_tokens": int(all_tokens.numel()),
        "timings": loop_timings,
        "per_step": {
            "ggml_wall_mean_ms": float(np.mean(ggml_wall_times) * 1000.0) if ggml_wall_times else None,
            "ggml_wall_min_ms": float(np.min(ggml_wall_times) * 1000.0) if ggml_wall_times else None,
            "ggml_wall_max_ms": float(np.max(ggml_wall_times) * 1000.0) if ggml_wall_times else None,
            "ggml_reported_mean_ms": float(np.mean(ggml_reported_ms)) if ggml_reported_ms else None,
            "ggml_reported_min_ms": float(np.min(ggml_reported_ms)) if ggml_reported_ms else None,
            "ggml_reported_max_ms": float(np.max(ggml_reported_ms)) if ggml_reported_ms else None,
        },
    }
    return all_tokens
