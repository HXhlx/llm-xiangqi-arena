"""Adapter for Spark-X2-Agent via LiteLLM / iFlytek MaaS.

LiteLLM model_info: thinking_param=enable_thinking (bool),
reasoning_effort_values=low|medium|high.
"""

from typing import Optional

from .base import BaseAdapter


class SparkX2Adapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        body: dict = {"enable_thinking": enable_thinking}
        if reasoning_effort:
            body["reasoning_effort"] = reasoning_effort
        return body
