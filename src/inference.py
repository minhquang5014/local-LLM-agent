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


def _apply_qwen3_chat_template(prompt: str) -> str:
    """
    Wrap a raw ReAct prompt in Qwen3 chat tokens.

    Without this, Qwen3 (a chat-tuned model) treats the input as a document
    continuation and ignores the system instructions — causing it to hallucinate
    instead of calling tools.  The empty <think>\\n\\n</think> block disables
    thinking mode, saving ~300-400 tokens of budget.
    """
    return (
        f"<|im_start|>user\n{prompt}<|im_end|>\n"
        f"<|im_start|>assistant\n"
        f"<think>\n\n</think>\n\n"
    )


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
    # Set per-request by the agent; checked between tokens so the Stop button
    # can abort a generation mid-stream instead of waiting for it to finish.
    _stop_event: Any = None

    class Config:
        arbitrary_types_allowed = True

    def _load(self):
        if self._model is None:
            from mlx_lm import load
            logger.info("Loading model with mlx-lm from %s …", self.model_path)
            self._model, self._tokenizer = load(self.model_path)
            logger.info("mlx-lm model loaded.")

    def set_stop_event(self, event) -> None:
        """Attach a threading.Event the agent can set to interrupt generation."""
        self._stop_event = event

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
        from mlx_lm import stream_generate
        self._load()
        from mlx_lm.sample_utils import make_sampler, make_repetition_penalty
        sampler = make_sampler(temp=self.temperature, top_p=self.top_p)
        logits_processors = [make_repetition_penalty(self.repetition_penalty)]
        formatted = _apply_qwen3_chat_template(prompt)

        # Stop sequences: cut as soon as one appears rather than generating to
        # max_tokens (faster), and the same list trims the final text.
        effective_stop = list(stop) if stop else []
        effective_stop += ["\nObservation:", "\nHuman:", "\nUser:"]

        stop_event = self._stop_event
        text = ""
        for resp in stream_generate(
            self._model,
            self._tokenizer,
            prompt=formatted,
            max_tokens=self.max_tokens,
            sampler=sampler,
            logits_processors=logits_processors,
        ):
            text += resp.text
            # User pressed Stop — abort generation immediately.
            if stop_event is not None and stop_event.is_set():
                break
            # Early-exit once a stop sequence appears (the model has finished the
            # useful part of this step — no need to keep generating).
            if any(s in text for s in effective_stop):
                break

        # Strip Qwen3 <think>...</think> blocks before any further processing.
        raw = _strip_thinking(text)
        # Fallback: also cut at the old "---" separator some versions used.
        raw = raw.split("\n---\n")[0].strip()
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


class HFChatLLM(LLM):
    """LangChain LLM wrapper around a HuggingFace text-generation pipeline.

    Applies the Qwen3 chat template before generation so the model respects
    system instructions rather than treating the prompt as a document continuation.
    """

    max_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.9
    repetition_penalty: float = 1.1

    _pipe: Any = None
    _stop_event: Any = None

    class Config:
        arbitrary_types_allowed = True

    def set_stop_event(self, event) -> None:
        """Attach a threading.Event the agent can set to interrupt generation."""
        self._stop_event = event

    @property
    def _llm_type(self) -> str:
        return "hf-chat"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        formatted = _apply_qwen3_chat_template(prompt)

        # Let the Stop button interrupt generation between tokens.
        stopping_criteria = None
        if self._stop_event is not None:
            from transformers import StoppingCriteria, StoppingCriteriaList

            event = self._stop_event

            class _EventStop(StoppingCriteria):
                def __call__(self, input_ids, scores, **kw):
                    return event.is_set()

            stopping_criteria = StoppingCriteriaList([_EventStop()])

        outputs = self._pipe(
            formatted,
            max_new_tokens=self.max_tokens,
            temperature=self.temperature,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            do_sample=True,
            return_full_text=False,
            stopping_criteria=stopping_criteria,
        )
        raw = outputs[0]["generated_text"] if outputs else ""
        raw = _strip_thinking(raw)
        # Honour stop sequences
        effective_stop = list(stop) if stop else []
        effective_stop += ["\nObservation:", "\nHuman:", "\nUser:"]
        for s in effective_stop:
            idx = raw.find(s)
            if idx != -1:
                raw = raw[:idx]
        return raw.strip()


def _build_hf_pipeline():
    import torch
    from transformers import AutoModelForCausalLM, PreTrainedTokenizerFast, pipeline
    from src.config import HF_MODEL_PATH, MAX_NEW_TOKENS, TEMPERATURE, TOP_P, REPETITION_PENALTY
    import os

    device = "mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Loading model via transformers on device=%s …", device)
    print(f"[INFERENCE] Loading HF model on device={device} from {HF_MODEL_PATH}")

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
    )
    llm = HFChatLLM(
        max_tokens=MAX_NEW_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        repetition_penalty=REPETITION_PENALTY,
    )
    llm._pipe = pipe
    return llm


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
