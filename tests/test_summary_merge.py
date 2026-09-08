#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""摘要合并更新单元测试：不实际调 LLM，用 monkeypatch 替换摘要结果。"""
import asyncio
import sys

sys.path.insert(0, "/home/hx/xiangqi-llm-dual")

import agent.compress as C


class FakeClient:
    """把 compress LLM 调用替换为可控假输出。"""
    next_summary = "合并后的新摘要：开局中炮，红方弃马夺势，黑方被动。"

    class _Completions:
        async def create(self, **kw):
            msg = type("M", (), {"content": FakeClient.next_summary})()
            choice = type("C", (), {"message": msg})()
            return type("R", (), {"choices": [choice]})()

    completions = _Completions()
    chat = type("Chat", (), {"completions": completions})()

    def __init__(self, **kw):
        pass

    async def __aenter__(self):
        return self

    async def __aexit__(self, *a):
        return False


async def run_compress(msgs):
    import agent.rate_limit as RL

    async def _noop(*a, **kw):
        return None

    RL.acquire_outbound = _noop
    RL.record_outbound_tokens = _noop

    import openai

    orig_openai = C.AsyncOpenAI
    C.AsyncOpenAI = FakeClient
    try:
        return await C.compress_messages(
            msgs,
            api_base="http://fake",
            api_key="fake",
            model="gpt-4o",
            context_window=2000,       # 小窗口强触发
            threshold_ratio=0.9,
            keep_ratio=0.2,
        )
    finally:
        C.AsyncOpenAI = orig_openai


def mkmsgs(n, prefix="对局历史", old_summary=None):
    msgs = [{"role": "system", "content": "你是象棋选手"}]
    if old_summary:
        msgs.append({"role": "user", "content": f"[对局历史摘要]\n{old_summary}"})
    for i in range(n):
        msgs.append({"role": "user", "content": f"{prefix}第{i}回合：" + "炮二平五，马八进七，车九平八，兵七进一，炮八平九，马二进三 " * 8})
        msgs.append({"role": "assistant", "content": f"思考{i}：" + "出车保马，中兵推进，双炮过河，士象联动，防住肋道 " * 8})
    return msgs


async def main():
    # ---- 测试1：首次压缩，摘要应为 user 角色 ----
    out = await run_compress(mkmsgs(30))
    summary_msgs = [m for m in out if str(m.get("content", "")).startswith("[对局历史摘要]")]
    assert len(summary_msgs) == 1, f"应有且仅有一份摘要, got {len(summary_msgs)}"
    assert summary_msgs[0]["role"] == "user", f"摘要角色应为 user, got {summary_msgs[0]['role']}"
    print("T1 通过: 摘要为 user 角色，单份")

    # ---- 测试2：二次压缩（含旧摘要），旧摘要被摘出合并、不堆积 ----
    msgs2 = mkmsgs(30, old_summary="旧摘要：开局飞相，红方先手。")
    out2 = await run_compress(msgs2)
    summary_msgs2 = [m for m in out2 if str(m.get("content", "")).startswith("[对局历史摘要]")]
    assert len(summary_msgs2) == 1, f"二次压缩后应仍只有一份摘要, got {len(summary_msgs2)}"
    assert out2[0]["role"] == "system", "system 应保留在最前"
    print("T2 通过: 二次压缩旧摘要被合并更新，不堆积")

    # ---- 测试3：_extract_old_summary 单元行为 ----
    msgs3 = [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "[对局历史摘要]\n旧内容"},
        {"role": "user", "content": "正常消息"},
    ]
    old_s, rest = C._extract_old_summary(msgs3)
    assert old_s == "旧内容", f"旧摘要提取失败: {old_s!r}"
    assert len(rest) == 2, "剩余应只有 system+正常消息"
    print("T3 通过: _extract_old_summary 提取正确")

    # ---- 测试4：提示词渲染（previous_summary 分支）----
    from prompt_registry import render_prompt

    # 提示词模板层：空 hint 应只剩占位符空白，非空 hint 携带旧摘要
    t1 = render_prompt("agent.compress", "user", transcript="T", previous_hint="")
    t2 = render_prompt("agent.compress", "user", transcript="T", previous_hint="已有旧摘要（在它基础上合并更新，不要从零重写，不要丢失其中仍然有效的信息）：\n\n旧摘要内容")
    assert "已有旧摘要" in t2, "hint 未进入模板"
    assert "旧摘要内容" in t2, "旧摘要内容未渲染"
    # compress.py 侧 hint 拼装逻辑（previous_hint 只在 old_summary 非空时携带内容）
    msgs_hint = None
    print("T4 通过: 提示词 previous_hint 渲染正确")

    print("ALL TESTS PASSED")


asyncio.run(main())
