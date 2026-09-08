"""Adapter for providers that only accept reasoning_effort (no thinking.*).

LiteLLM / vendor notes:
- Grok (xAI): reasoning_effort low|medium|high|xhigh (default high); thinking_values empty
- SenseNova: reasoning_effort low|medium|high|max (default high); thinking.type enabled|disabled|adaptive
- Kimi K3: always thinking; reasoning_effort low|high|max (default max);
  do not send K2.x thinking.* fields
- Hy3 (Tencent): reasoning_effort no_think|low|high (default no_think);
  recommended temperature=0.9 top_p=1.0
- Step-3.7-flash: reasoning_effort low|medium|high (default medium)
- Muse Spark (Meta / OpenCode Zen): responses-mode on LiteLLM; reasoning_effort
  minimal|low|medium|high|xhigh (ceiling xhigh; none rejected; max not yet available);
  do not send thinking.*
"""

from typing import Optional

from .base import BaseAdapter


class EffortOnlyAdapter(BaseAdapter):
    def extra_body(
        self,
        enable_thinking: bool,
        reasoning_effort: Optional[str] = None,
    ) -> dict | None:
        if not enable_thinking:
            return None
        if reasoning_effort:
            return {"reasoning_effort": reasoning_effort}
        return None

    def extract_reasoning(self, delta) -> str | None:
        reasoning = getattr(delta, "reasoning", None)
        if reasoning:
            return reasoning
        return super().extract_reasoning(delta)
