"""Adapter for Spark / XF-Yun models."""

from typing import Optional

from .base import BaseAdapter

# Xunfei MaaS OpenAI-compatible calls expect lora_id=0 for base models
# (replace with resourceId when calling a fine-tuned service).
_LORA_ID_HEADER = {"lora_id": "0"}


class SparkAdapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        body: dict = {
            "thinking": {"type": "enabled" if enable_thinking else "disabled"},
        }
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return body

    def patch_request_args(self, request_args: dict) -> dict:
        headers = dict(request_args.get("extra_headers") or {})
        headers.update(_LORA_ID_HEADER)
        return {**request_args, "extra_headers": headers}
