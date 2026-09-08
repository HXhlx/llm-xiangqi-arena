"""LangGraph LLM player with Redis Stack checkpointer + context compression."""

from __future__ import annotations

import asyncio
import json
import time
from typing import Any, AsyncIterator, Optional

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph
from openai import APIConnectionError, APIStatusError, AsyncOpenAI

from adapters import get_adapter
from agent.observe_session import ObserveSession, is_submit_ok
from llm_client import (
    DEFAULT_MAX_TOOL_ROUNDS,
    TOOL_DEFINITIONS,
    build_system_prompt,
    execute_tool,
    _turn_prompt,
)
from xiangqi import Board

from .checkpoint_sync import merge_model_messages
from .compress import (
    checkpoint_messages_need_sync,
    compress_messages,
    estimate_text_tokens,
    estimate_tokens,
    hard_trim,
    should_compress,
    without_orphan_tools,
)
from .rate_limit import acquire_outbound, record_outbound_tokens
from .graph import AgentState
from . import memory as mem
from .memory_archive import try_write_pre_compress_snapshot
from .turn_store import format_inject_block, load_last_turn


REASONING_EFFORTS = frozenset(
    {"none", "minimal", "low", "medium", "high", "xhigh", "max"}
)
MAX_MAKE_MOVE_NUDGES = 3

_MOVE_GATE_PHASES = {
    "waiting": "预盘/开局前",
    "paused": "复盘/暂停",
    "finished": "终局复盘",
    "interrupted": "中断",
}


def move_gate_error(status: str | None) -> Optional[str]:
    """走子阶段闸门：非 playing 时返回禁用错误文本，否则 None。

    与 MCP 侧 submit_move 的阶段闸门保持一致：复盘/预盘/终局/中断
    阶段 make_move 一律禁用，防止对局停止后 agent 仍提交着法。
    """
    st = (status or "").strip().lower()
    if not st or st == "playing":
        return None
    phase = _MOVE_GATE_PHASES.get(st, st)
    return (
        f"make_move 已禁用（对局 status={st}，{phase}阶段不走子）。"
        "此着不会被受理；对局恢复 playing 且轮到你时再提交。"
    )


def should_count_missing_make_move(tool_names: list[str] | None) -> bool:
    """Count a miss only when the model produced no tool calls (plain text)."""
    return not bool(tool_names)


def _current_task_cancelling() -> bool:
    """True when this task was cancelled (3.11+). Catching CancelledError
    from a child task must not clear that flag or run later cleanup."""
    me = asyncio.current_task()
    if me is None:
        return False
    cancelling = getattr(me, "cancelling", None)
    return bool(cancelling()) if callable(cancelling) else False


def handle_missing_make_move(misses_so_far: int) -> dict[str, Any]:
    """No make_move this model round: perceivable tool-node error, at most 3 times."""
    nxt = int(misses_so_far or 0) + 1
    if nxt > MAX_MAKE_MOVE_NUDGES:
        return {
            "done": True,
            "move": None,
            "error": (
                f"Failed to get a valid move: make_move was not called after "
                f"{MAX_MAKE_MOVE_NUDGES} prompts."
            ),
            "make_move_misses": nxt,
        }
    return {
        "done": False,
        "move": None,
        "make_move_misses": nxt,
        "error_text": (
            f"工具节点错误 [{nxt}/{MAX_MAKE_MOVE_NUDGES}]：本回合未调用 make_move。"
            "必须使用 make_move 提交真实盘走法（ICCS，例如 h2e2）。正文里的坐标不会落子。"
        ),
    }


def make_thread_id(game_id: str, side_name: str) -> str:
    prefix = mem.redis_prefix()
    return f"{prefix}{game_id}:{side_name}"


async def clear_player_thread(game_id: str, side_name: str) -> None:
    await mem.delete_thread(make_thread_id(game_id, side_name))


async def clear_game_threads(game_id: str) -> None:
    await mem.delete_thread(make_thread_id(game_id, "red"))
    await mem.delete_thread(make_thread_id(game_id, "black"))


def _msg_to_openai_dict(msg) -> dict[str, Any]:
    """Convert LangChain message (or dict) to OpenAI chat dict."""
    if isinstance(msg, dict):
        return dict(msg)
    role = getattr(msg, "type", None) or getattr(msg, "role", None)
    role_map = {
        "human": "user",
        "ai": "assistant",
        "system": "system",
        "tool": "tool",
    }
    out_role = role_map.get(role, role or "user")
    content = getattr(msg, "content", "") or ""
    d: dict[str, Any] = {"role": out_role, "content": content if isinstance(content, str) else str(content)}
    if out_role == "assistant":
        tool_calls = getattr(msg, "tool_calls", None) or []
        if tool_calls:
            d["tool_calls"] = []
            for tc in tool_calls:
                if isinstance(tc, dict):
                    # LC format: {id, name, args}
                    args = tc.get("args", tc.get("arguments", {}))
                    if not isinstance(args, str):
                        args = json.dumps(args or {}, ensure_ascii=False)
                    d["tool_calls"].append(
                        {
                            "id": tc.get("id") or "call_0",
                            "type": "function",
                            "function": {
                                "name": tc.get("name") or tc.get("function", {}).get("name", ""),
                                "arguments": args
                                if isinstance(args, str)
                                else json.dumps(args or {}, ensure_ascii=False),
                            },
                        }
                    )
                else:
                    args = getattr(tc, "args", {}) or {}
                    d["tool_calls"].append(
                        {
                            "id": getattr(tc, "id", None) or "call_0",
                            "type": "function",
                            "function": {
                                "name": getattr(tc, "name", "") or "",
                                "arguments": json.dumps(args, ensure_ascii=False)
                                if not isinstance(args, str)
                                else args,
                            },
                        }
                    )
        # additional_kwargs tool_calls (OpenAI style)
        ak = getattr(msg, "additional_kwargs", None) or {}
        if not d.get("tool_calls") and ak.get("tool_calls"):
            d["tool_calls"] = ak["tool_calls"]
    if out_role == "tool":
        d["tool_call_id"] = getattr(msg, "tool_call_id", None) or ""
        name = getattr(msg, "name", None)
        if name:
            d["name"] = name
    return d


def _openai_dicts_from_state_messages(messages: list) -> list[dict[str, Any]]:
    return [_msg_to_openai_dict(m) for m in messages]


class LangGraphPlayer:
    """Stateful LLM player backed by LangGraph + Redis Stack checkpointer."""

    def __init__(
        self,
        api_base: str,
        api_key: str,
        model: str,
        *,
        game_id: str = "",
        side_name: str = "red",
        timeout: Optional[float] = None,
        max_tool_rounds: int = DEFAULT_MAX_TOOL_ROUNDS,
        prompt_name: str = "zh",
        enable_thinking: bool = True,
        max_completion_tokens: int = 8192,
        context_window: int = 256000,
        max_input_tokens: Optional[int] = None,
        compress_threshold_ratio: float = 0.9,
        compress_model: Optional[str] = None,
        enable_reasoning_effort: bool = False,
        reasoning_effort: Optional[str] = None,
        preset: Optional[str] = None,
        rpm: Optional[int] = None,
        tpm: Optional[int] = None,
        status_probe: Optional[Any] = None,
    ):
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.preset = (preset or "").strip() or model
        self.rpm = rpm
        self.tpm = tpm
        self.game_id = game_id
        self.side_name = side_name
        self.timeout = timeout
        self.max_tool_rounds = max_tool_rounds
        self.prompt_name = prompt_name
        self.enable_thinking = enable_thinking
        self.max_completion_tokens = max_completion_tokens
        self.context_window = max(4096, int(context_window or 256000))
        if max_input_tokens is None:
            self.max_input_tokens = self.context_window
        else:
            self.max_input_tokens = max(
                4096, min(self.context_window, int(max_input_tokens))
            )
        self.compress_threshold_ratio = float(compress_threshold_ratio or 0.9)
        self.compress_model = (compress_model or model).strip() or model
        self.enable_reasoning_effort = bool(enable_reasoning_effort)
        effort = (reasoning_effort or "").strip().lower() or None
        if effort and effort not in REASONING_EFFORTS:
            effort = None
        self.reasoning_effort = effort if self.enable_reasoning_effort else None
        self.adapter = get_adapter(api_base, model)
        self.thread_id = make_thread_id(game_id or "nogame", side_name or "red")
        self._archived_this_turn = False
        self._board: Optional[Board] = None
        self._observe: Optional[ObserveSession] = None
        self._side: str = "w"
        self._live_queue: Optional[asyncio.Queue] = None
        self._turn_trace: dict[str, Any] = self._empty_turn_trace()
        self.last_turn_trace: Optional[dict[str, Any]] = None
        # 走子阶段闸门：回调返回对局 status；非 playing 时 make_move 禁用
        self._status_probe = status_probe

    @staticmethod
    def _empty_turn_trace() -> dict[str, Any]:
        return {
            "reasoning": "",
            "thinking": "",
            "assistant_content": "",
            "tool_rounds": [],
        }

    def _build_request_args(self, messages: list[dict], use_tools: bool = True) -> dict:
        request_args: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            "stream": True,
            "max_completion_tokens": self.max_completion_tokens,
        }
        if use_tools:
            request_args["tools"] = TOOL_DEFINITIONS
            request_args["tool_choice"] = "auto" if self.enable_thinking else "required"

        extra_body = self.adapter.extra_body(
            self.enable_thinking,
            reasoning_effort=self.reasoning_effort,
        )
        if extra_body:
            request_args["extra_body"] = extra_body
        return self.adapter.patch_request_args(request_args)

    async def _stream_completion(self, oai_messages: list[dict]):
        request_args = self._build_request_args(oai_messages, use_tools=True)
        accumulated_content = ""
        accumulated_reasoning = ""
        tool_calls_acc: dict[int, dict] = {}
        finish_reason = None
        # Live thought push: ~1 event/sec carrying NEW text since last emit.
        # Frontend appends these chunks so the full transcript is shown.
        pending_reasoning = ""
        pending_thinking = ""
        last_emit = 0.0
        emit_interval = 1.0
        # Soft cap per tick to avoid huge SSE frames; overflow stays queued.
        max_emit_chars = 4000

        def _put_event(payload: dict) -> None:
            q = self._live_queue
            if q is None:
                return
            try:
                q.put_nowait(("event", payload))
            except asyncio.QueueFull:
                # Drop one oldest event item if full, then retry once.
                try:
                    q.get_nowait()
                except asyncio.QueueEmpty:
                    return
                try:
                    q.put_nowait(("event", payload))
                except asyncio.QueueFull:
                    pass

        def emit_thought(force: bool = False) -> None:
            nonlocal pending_reasoning, pending_thinking, last_emit
            now = time.monotonic()
            if not force and (now - last_emit) < emit_interval:
                return
            if not pending_reasoning and not pending_thinking:
                if force:
                    last_emit = now
                return
            # Emit chronological NEW text since last tick (frontend appends).
            if force:
                r_chunk = pending_reasoning
                t_chunk = pending_thinking
                pending_reasoning = ""
                pending_thinking = ""
            else:
                r_chunk = pending_reasoning[:max_emit_chars] if pending_reasoning else ""
                t_chunk = pending_thinking[:max_emit_chars] if pending_thinking else ""
                pending_reasoning = pending_reasoning[len(r_chunk):]
                pending_thinking = pending_thinking[len(t_chunk):]
            if not r_chunk and not t_chunk:
                return
            last_emit = now
            _put_event(
                {
                    "type": "thought",
                    "mode": "append",
                    "reasoning": r_chunk,
                    "thinking": t_chunk,
                }
            )

        client_kwargs: dict[str, Any] = {
            "api_key": self.api_key,
            "base_url": self.api_base,
        }
        if self.timeout is not None:
            client_kwargs["timeout"] = self.timeout
        await acquire_outbound(
            self.preset,
            tokens=estimate_tokens(oai_messages, self.model),
            rpm=self.rpm,
            tpm=self.tpm,
        )
        async with AsyncOpenAI(**client_kwargs) as client:
            stream = await client.chat.completions.create(**request_args)
            async for chunk in stream:
                choices = getattr(chunk, "choices", None) or []
                if not choices:
                    emit_thought(False)
                    continue
                choice = choices[0]
                delta = getattr(choice, "delta", None)
                if not delta:
                    emit_thought(False)
                    continue
                fr = getattr(choice, "finish_reason", None)
                if fr:
                    finish_reason = fr

                reasoning_piece = self.adapter.extract_reasoning(delta)
                if reasoning_piece:
                    pending_reasoning += reasoning_piece
                    accumulated_reasoning += reasoning_piece

                content_piece = getattr(delta, "content", None)
                if content_piece:
                    accumulated_content += content_piece
                    pending_thinking += content_piece

                for tcd in getattr(delta, "tool_calls", None) or []:
                    idx = getattr(tcd, "index", 0) or 0
                    if idx not in tool_calls_acc:
                        tool_calls_acc[idx] = {
                            "id": getattr(tcd, "id", None) or f"call_{idx}",
                            "name": "",
                            "arguments": "",
                        }
                    if getattr(tcd, "id", None):
                        tool_calls_acc[idx]["id"] = tcd.id
                    func = getattr(tcd, "function", None)
                    if func and getattr(func, "name", None):
                        tool_calls_acc[idx]["name"] += func.name
                    if func and getattr(func, "arguments", None):
                        tool_calls_acc[idx]["arguments"] += func.arguments

                emit_thought(False)

        emit_thought(force=True)
        out_tokens = estimate_text_tokens(
            accumulated_content + accumulated_reasoning, self.model
        )
        if out_tokens:
            await record_outbound_tokens(self.preset, out_tokens, tpm=self.tpm)

        tool_calls = None
        if tool_calls_acc:
            tool_calls = []
            for idx in sorted(tool_calls_acc.keys()):
                tc = tool_calls_acc[idx]
                tool_calls.append(
                    {
                        "id": tc["id"],
                        "name": tc["name"],
                        "args": tc["arguments"],
                    }
                )
        return accumulated_content, tool_calls, finish_reason or "stop", accumulated_reasoning

    async def _maybe_compress_openai_messages(self, oai_msgs: list[dict]) -> list[dict]:
        # Compress against max_input_tokens (defaults to context_window).
        budget = self.max_input_tokens
        if should_compress(
            oai_msgs,
            budget,
            self.compress_threshold_ratio,
            self.model,
        ):
            try:
                oai_msgs = await compress_messages(
                    oai_msgs,
                    api_base=self.api_base,
                    api_key=self.api_key,
                    model=self.compress_model,
                    context_window=budget,
                    threshold_ratio=self.compress_threshold_ratio,
                    timeout=self.timeout if self.timeout is not None else None,
                    preset=self.preset,
                    rpm=self.rpm,
                    tpm=self.tpm,
                )
            except Exception:
                oai_msgs = hard_trim(oai_msgs, budget, model=self.model)
        return oai_msgs

    def _build_graph(self, board: Board, side: str):
        player = self

        async def prepare_turn(state: AgentState) -> dict:
            system_prompt = build_system_prompt(board, side, player.prompt_name)
            turn_text = _turn_prompt(board, side, player.prompt_name)
            existing = list(state.get("messages") or [])
            updates: list = []

            # Inject previous own-side full thought (from JSONL) before board snapshot.
            prev = load_last_turn(player.game_id, player.side_name)
            if prev:
                updates.append(HumanMessage(content=format_inject_block(prev)))

            # First turn: full system prompt. Later turns: only board-updated user turn
            # to avoid unbounded system prompt duplication in checkpointed history.
            if not existing:
                updates.append(SystemMessage(content=system_prompt))
                updates.append(HumanMessage(content=turn_text))
            else:
                board_update = (
                    f"{turn_text}\n\n"
                    f"[局面更新]\n{system_prompt}"
                )
                updates.append(HumanMessage(content=board_update))
            player._observe = ObserveSession(board)
            # Reset per-ply trace for this request_move invocation
            player._turn_trace = player._empty_turn_trace()
            return {
                "messages": updates,
                "side": side,
                "fen": board.to_fen(),
                "move": None,
                "error": None,
                "done": False,
                "make_move_misses": 0,
            }

        async def call_model(state: AgentState) -> dict:
            raw_oai = _openai_dicts_from_state_messages(list(state.get("messages") or []))
            # Drop duplicate consecutive systems except summary ones: keep last system before last human
            oai_msgs = await player._maybe_compress_openai_messages(raw_oai)
            replace = False
            cleaned = without_orphan_tools(oai_msgs)
            if checkpoint_messages_need_sync(raw_oai, cleaned):
                fen = str(state.get("fen") or (player._board.to_fen() if player._board else ""))
                replace = try_write_pre_compress_snapshot(
                    player.game_id,
                    player.side_name,
                    fen=fen,
                    messages=raw_oai,
                )
                if replace:
                    player._archived_this_turn = True

            def pack(new_msgs: list | None = None, **extra: Any) -> dict:
                synced = merge_model_messages(raw_oai, oai_msgs, new_msgs, replace=replace)
                if synced:
                    extra["messages"] = synced
                return extra

            max_api_retries = 3
            last_error_msg = ""
            content, tool_calls, finish_reason, reasoning_full = "", None, None, ""
            for attempt in range(1, max_api_retries + 2):  # 1 + 3 retries
                try:
                    content, tool_calls, finish_reason, reasoning_full = await player._stream_completion(oai_msgs)
                    break
                except APIStatusError as e:
                    last_error_msg = f"API HTTP error: {e.status_code} - {str(e)[:200]}"
                except APIConnectionError as e:
                    last_error_msg = f"API connection error: {str(e)[:200]}"
                except TimeoutError as e:
                    last_error_msg = f"API timeout: {str(e)[:200]}"
                except Exception as e:
                    last_error_msg = f"API error: {str(e)[:200]}"
                if attempt <= max_api_retries:
                    print(f"  [{side}] API attempt {attempt}/{max_api_retries+1} failed: {last_error_msg}; retrying in 5s...")
                    await asyncio.sleep(5)
                else:
                    print(f"  [{side}] API exhausted {max_api_retries+1} attempts: {last_error_msg}")
            else:
                return pack(error=last_error_msg, done=True)

            # Accumulate into turn_trace (full text for JSONL / next inject)
            if reasoning_full:
                player._turn_trace["reasoning"] = (
                    (player._turn_trace.get("reasoning") or "") + reasoning_full
                )
            if content:
                player._turn_trace["assistant_content"] = (
                    (player._turn_trace.get("assistant_content") or "") + content
                )
                player._turn_trace["thinking"] = (
                    (player._turn_trace.get("thinking") or "") + content
                )

            if tool_calls:
                lc_tool_calls = []
                oai_tool_calls = []
                pending_round_calls = []
                for tc in tool_calls:
                    raw_args = tc.get("args") or "{}"
                    try:
                        args_obj = json.loads(raw_args) if isinstance(raw_args, str) else (raw_args or {})
                    except json.JSONDecodeError:
                        args_obj = {}
                    lc_tool_calls.append(
                        {
                            "id": tc.get("id") or "call_0",
                            "name": tc.get("name") or "",
                            "args": args_obj,
                            "type": "tool_call",
                        }
                    )
                    oai_tool_calls.append(
                        {
                            "id": tc.get("id") or "call_0",
                            "type": "function",
                            "function": {
                                "name": tc.get("name") or "",
                                "arguments": raw_args
                                if isinstance(raw_args, str)
                                else json.dumps(raw_args or {}, ensure_ascii=False),
                            },
                        }
                    )
                    pending_round_calls.append(
                        {"name": tc.get("name") or "", "args": args_obj, "id": tc.get("id") or "call_0"}
                    )
                # Stash for execute_tools to pair with results
                player._turn_trace["_pending_tool_calls"] = pending_round_calls

                ai = AIMessage(
                    content=content or "",
                    tool_calls=lc_tool_calls,
                    additional_kwargs={"tool_calls": oai_tool_calls},
                )
                return pack([ai], done=False)

            ai = AIMessage(content=content or "")
            nudge = handle_missing_make_move(int(state.get("make_move_misses") or 0))
            if nudge.get("done"):
                player.last_turn_trace = dict(player._turn_trace)
                return pack(
                    [ai],
                    done=True,
                    error=nudge.get("error"),
                    make_move_misses=nudge["make_move_misses"],
                )
            retry = HumanMessage(content=nudge["error_text"])
            return pack(
                [ai, retry],
                done=False,
                make_move_misses=nudge["make_move_misses"],
            )

        async def execute_tools(state: AgentState) -> dict:
            messages = list(state.get("messages") or [])
            last = messages[-1] if messages else None
            tool_calls = getattr(last, "tool_calls", None) or []
            if not tool_calls:
                return {"done": False}

            tool_messages = []
            found_move = None
            round_calls: list[dict] = []
            round_results: list[dict] = []
            pending = player._turn_trace.pop("_pending_tool_calls", None) or []
            session = player._observe or ObserveSession(board)
            player._observe = session

            for i, tc in enumerate(tool_calls):
                if isinstance(tc, dict):
                    name = tc.get("name") or ""
                    args = tc.get("args") or {}
                    tc_id = tc.get("id") or "call_0"
                else:
                    name = getattr(tc, "name", "") or ""
                    args = getattr(tc, "args", {}) or {}
                    tc_id = getattr(tc, "id", None) or "call_0"

                if i < len(pending):
                    round_calls.append(pending[i])
                else:
                    round_calls.append({"name": name, "args": args, "id": tc_id})

                # 走子阶段闸门：非 playing（复盘/预盘/终局/中断）禁用 make_move
                gate_err = None
                if name == "make_move" and player._status_probe is not None:
                    try:
                        probe_status = player._status_probe()
                    except Exception:
                        probe_status = None
                    gate_err = move_gate_error(probe_status)
                if gate_err is not None:
                    result = gate_err
                else:
                    result = execute_tool(session, name, args if isinstance(args, dict) else {})
                tool_messages.append(
                    ToolMessage(content=result, tool_call_id=tc_id, name=name)
                )
                round_results.append({"name": name, "content": result})
                if name == "make_move" and is_submit_ok(result):
                    found_move = (args.get("move") if isinstance(args, dict) else "") or ""
                    found_move = str(found_move).strip().lower()

            player._turn_trace.setdefault("tool_rounds", []).append(
                {"tool_calls": round_calls, "tool_results": round_results}
            )

            out: dict[str, Any] = {"messages": tool_messages}
            # 闸门触发：对局已非 playing，立即终止本回合（不再消耗模型轮次）
            if gate_err is not None:
                player.last_turn_trace = dict(player._turn_trace)
                out["done"] = True
                out["aborted"] = True
                return out
            if found_move:
                out["move"] = found_move
                out["done"] = True
                player.last_turn_trace = dict(player._turn_trace)
                return out
            names = [
                (tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")) or ""
                for tc in tool_calls
            ]
            if not should_count_missing_make_move(names):
                out["done"] = False
                return out
            nudge = handle_missing_make_move(int(state.get("make_move_misses") or 0))
            if nudge.get("done"):
                player.last_turn_trace = dict(player._turn_trace)
                out["done"] = True
                out["error"] = nudge.get("error")
                out["make_move_misses"] = nudge["make_move_misses"]
                return out
            out["done"] = False
            out["make_move_misses"] = nudge["make_move_misses"]
            out["messages"] = tool_messages + [HumanMessage(content=nudge["error_text"])]
            return out

        def route_after_model(state: AgentState) -> str:
            if state.get("error") or state.get("done"):
                return "end"
            messages = list(state.get("messages") or [])
            last = messages[-1] if messages else None
            if last is not None and getattr(last, "tool_calls", None):
                return "tools"
            if state.get("done"):
                return "end"
            # retry loop via model again
            return "model"

        def route_after_tools(state: AgentState) -> str:
            if state.get("done") and state.get("move"):
                return "end"
            if state.get("error"):
                return "end"
            if state.get("aborted"):
                return "end"
            return "model"

        g = StateGraph(AgentState)
        g.add_node("prepare", prepare_turn)
        g.add_node("model", call_model)
        g.add_node("tools", execute_tools)
        g.add_edge(START, "prepare")
        g.add_edge("prepare", "model")
        g.add_conditional_edges(
            "model",
            route_after_model,
            {"tools": "tools", "model": "model", "end": END},
        )
        g.add_conditional_edges(
            "tools",
            route_after_tools,
            {"model": "model", "end": END},
        )
        return g

    async def request_move(self, board: Board, side: str) -> AsyncIterator[dict]:
        self._board = board
        self._side = side
        checkpointer = await mem.get_checkpointer()
        graph = self._build_graph(board, side).compile(checkpointer=checkpointer)
        config = {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": max(400, self.max_tool_rounds * 2),
        }

        live_q: asyncio.Queue = asyncio.Queue(maxsize=8)
        self._live_queue = live_q
        self._archived_this_turn = False
        final_move = None
        final_error = None
        final_aborted = False

        async def _run():
            try:
                result = await graph.ainvoke(
                    {
                        "side": side,
                        "fen": board.to_fen(),
                        "move": None,
                        "error": None,
                        "done": False,
                    },
                    config,
                )
                await live_q.put(("done", result if isinstance(result, dict) else {}))
            except asyncio.CancelledError:
                raise
            except Exception as e:
                await live_q.put(("error", str(e)[:200]))

        task = asyncio.create_task(_run())
        try:
            while True:
                kind, payload = await live_q.get()
                if kind == "event":
                    yield payload
                elif kind == "done":
                    final_move = payload.get("move")
                    final_error = payload.get("error")
                    final_aborted = bool(payload.get("aborted"))
                    break
                elif kind == "error":
                    yield {"type": "error", "message": f"Graph error: {payload}"}
                    return
        finally:
            self._live_queue = None
            if not task.done():
                task.cancel()
                try:
                    await asyncio.wait_for(task, timeout=5)
                except asyncio.CancelledError:
                    if _current_task_cancelling():
                        raise
                except Exception:
                    pass
            if _current_task_cancelling():
                raise asyncio.CancelledError
            try:
                await asyncio.wait_for(
                    mem.prune_thread_to_latest(self.thread_id),
                    timeout=8,
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                print(f"  [memory] prune failed {self.thread_id}: {exc}")

        if final_move:
            yield {"type": "move", "move": final_move}
            return
        if final_aborted:
            # 走子阶段闸门触发：对局已非 playing（复盘/预盘/终局/中断），
            # 本回合被终止且不视为模型失败 —— 不计错误、不触发 _interrupt_game
            yield {"type": "aborted", "reason": "game_not_playing"}
            return
        if final_error:
            yield {"type": "error", "message": final_error}
            return
        yield {
            "type": "error",
            "message": f"Failed to get a valid move after {self.max_tool_rounds} rounds.",
        }
