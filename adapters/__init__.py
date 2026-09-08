"""Model adapter registry. Maps (api_base, model) to the appropriate adapter."""

from .base import BaseAdapter
from .chat_template_thinking import ChatTemplateThinkingAdapter
from .default import DefaultAdapter
from .doubao import DoubaoAdapter
from .effort_only import EffortOnlyAdapter
from .gemini import GeminiAdapter
from .minimax import MiniMaxAdapter
from .openrouter import OpenRouterAdapter
from .qwen import QwenAdapter
from .spark import SparkAdapter
from .spark_x2 import SparkX2Adapter

_REGISTRY: list[tuple[callable, type[BaseAdapter]]] = [
    # Spark-X2-Agent: enable_thinking bool (before generic spark / xf-yun)
    (
        lambda api, model: model.startswith("spark-x2") or model.endswith("x2-agent"),
        SparkX2Adapter,
    ),
    (lambda api, model: "xf-yun.com" in api, SparkAdapter),
    (lambda api, model: model.startswith("spark"), SparkAdapter),
    (lambda api, model: "volces.com" in api, DoubaoAdapter),
    (lambda api, model: model.startswith("doubao"), DoubaoAdapter),
    (lambda api, model: "dashscope" in api, QwenAdapter),
    (lambda api, model: "openrouter.ai" in api, OpenRouterAdapter),
    (lambda api, model: model.startswith("gemini"), GeminiAdapter),
    # MiniMax-M3: thinking.type adaptive|disabled; drop reasoning_effort
    (lambda api, model: "minimax" in model, MiniMaxAdapter),
    # Dots3-Note: chat_template_kwargs.enable_thinking (no thinking.* / effort)
    (
        lambda api, model: model.startswith("dots3") or model.startswith("dots-3"),
        ChatTemplateThinkingAdapter,
    ),
    # Grok / SenseNova / Kimi / Hy3 / Step / Muse: reasoning_effort only (no thinking.*)
    # Grok: low|medium|high|xhigh | SenseNova: low|medium|high|max | Kimi: low|high|max
    # Hy3: no_think|low|high | Step: low|medium|high
    # Muse Spark: minimal|low|medium|high|xhigh (LiteLLM mode=responses)
    (lambda api, model: model.startswith("grok"), EffortOnlyAdapter),
    (lambda api, model: model.startswith("sensenova"), EffortOnlyAdapter),
    (lambda api, model: model.startswith("kimi"), EffortOnlyAdapter),
    (lambda api, model: model.startswith("hy3"), EffortOnlyAdapter),
    (lambda api, model: model.startswith("step-"), EffortOnlyAdapter),
    (
        lambda api, model: model.startswith("muse-spark") or model.startswith("muse_spark"),
        EffortOnlyAdapter,
    ),
]


def get_adapter(api_base: str, model: str) -> BaseAdapter:
    """Return the appropriate adapter for a given (api_base, model) pair."""
    api_lower = (api_base or "").lower()
    model_lower = (model or "").lower()
    for predicate, adapter_cls in _REGISTRY:
        if predicate(api_lower, model_lower):
            return adapter_cls()
    return DefaultAdapter()
