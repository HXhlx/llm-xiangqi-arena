"""LangGraph definition for a single Xiangqi LLM side (checkpointed messages)."""

from __future__ import annotations

from typing import Annotated, Any, Optional, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages


class AgentState(TypedDict, total=False):
    messages: Annotated[list, add_messages]
    side: str
    fen: str
    move: Optional[str]
    error: Optional[str]
    # Ephemeral control flags for one invocation (not critical to persist long-term)
    done: bool
    make_move_misses: int
    # 走子阶段闸门触发：对局已非 playing（复盘/预盘/终局/中断），
    # 本回合被终止且不视为模型失败
    aborted: bool


def build_base_graph():
    """Return an uncompiled StateGraph; nodes are attached by the player."""
    return StateGraph(AgentState)
