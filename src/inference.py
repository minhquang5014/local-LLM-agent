"""
Local inference backend.

Priority order:
  1. Ollama  — if the server is running (auto-detected)
  2. mlx-lm  — Apple Silicon native, loads the OptiQ safetensors model directly (~40 tok/s)
  3. HuggingFace transformers — fallback for non-Apple or non-MLX environments

Usage:
    from src.inference import get_llm
    llm = get_llm()
    response = llm.invoke("What is 2+2?")
"""

from __future__ import annotations

import logging
import re
from typing import Any, Iterator, List, Optional

from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun

logger = logging.getLogger(__name__)

_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)


def _strip_thinking(text: str) -> str:
    """Remove Qwen3 <think>...</think> blocks from raw model output."""
    return _THINK_RE.sub("", text).strip()


# ------------------------------------------------------------------
# Custom LangChain wrapper for mlx-lm
# ------------------------------------------------------------------

class MlxLM(LLM):
    """LangChain LLM wrapper around mlx-lm generate()."""

    model_path: str
    max_tokens: int = 512
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    # These are set after load — excluded from pydantic schema
    _model: Any = None
    _tokenizer: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _load(self):
        if self._model is None:
            from mlx_lm import load
            logger.info("Loading model with mlx-lm from %s …", self.model_path)
            self._model, self._tokenizer = load(self.model_path)
            logger.info("mlx-lm model loaded.")

    @property
    def _llm_type(self) -> str:
        return "mlx-lm"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        from mlx_lm import generate as mlx_generate
        self._load()
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        logits_processors = [make_repetition_penalty(self.repetition_penalty)]
        raw = mlx_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
            verbose=False,
        )
        # Strip Qwen3 <think>...</think> blocks before any further processing.
        raw = _strip_thinking(raw)
        # Fallback: also cut at the old "---" separator some versions used.
        raw = raw.split("\n---\n")[0].strip()
        # Honour stop sequences — mlx-lm may not support them natively, so we
        # post-process here.  This is the primary defence against the model
        # hallucinating fake Observation: blocks.
        effective_stop = list(stop) if stop else []
        effective_stop += ["\nObservation:", "\nHuman:", "\nUser:"]
        for s in effective_stop:
            idx = raw.find(s)
            if idx != -1:
                raw = raw[:idx]
        return raw.strip()


# ------------------------------------------------------------------
# Backend helpers
# ------------------------------------------------------------------

def _try_ollama():
    try:
        import requests
        from src.config import OLLAMA_BASE_URL, OLLAMA_MODEL
        r = requests.get(f"{OLLAMA_BASE_URL}/api/tags", timeout=2)
        r.raise_for_status()
        from langchain_ollama import OllamaLLM
        logger.info("Ollama detected — using OllamaLLM (%s)", OLLAMA_MODEL)
        return OllamaLLM(base_url=OLLAMA_BASE_URL, model=OLLAMA_MODEL)
    except Exception:
        return None


def _build_mlx():
    from src.config import HF_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
    llm = MlxLM(
        model_path=HF_MODEL_PATH,
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )
    llm._load()
    return llm


def _build_hf_pipeline():
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, pipeline
    from langchain_community.llms import HuggingFacePipeline
    from src.config import HF_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
    import os

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model via transformers on device=%s …", device)

    tokenizer = PreTrainedTokenizerFast(
        tokenizer_file=os.path.join(HF_MODEL_PATH, "tokenizer.json"),
        tokenizer_config_file=os.path.join(HF_MODEL_PATH, "tokenizer_config.json"),
    )
    tokenizer.eos_token = "<|im_end|>"
    tokenizer.pad_token = "<|endoftext|>"

    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        dtype=torch.float16,
        device_map=device,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
        do_sample=True,
        return_full_text=False,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------

_llm_instance = None


def get_llm() -> LLM:
    """Return a LangChain-compatible LLM (cached singleton).

    Auto-selects: Ollama → mlx-lm → transformers.
    """
    global _llm_instance
    if _llm_instance is not None:
        return _llm_instance

    llm = _try_ollama()
    if llm is not None:
        _llm_instance = llm
        return _llm_instance

    try:
        import mlx_lm  # noqa: F401
        _llm_instance = _build_mlx()
        return _llm_instance
    except ImportError:
        pass

    _llm_instance = _build_hf_pipeline()
    return _llm_instance


def generate(prompt: str) -> str:
    """Convenience wrapper: generate text from a plain string prompt."""
    return get_llm().invoke(prompt)
