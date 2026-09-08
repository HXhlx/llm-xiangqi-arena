"""Per-turn working-board cursor for LLM observe tools."""

from __future__ import annotations

from cycle_rules import is_threefold_after, occurrence_after
from agent.observe import (
    _ICCS,
    _color_zh,
    _cycle_kind_zh,
    board_diagram,
    legal_moves_annotated,
    threats,
    work_turn_line,
)
from xiangqi import PIECE_NAMES_ZH, Board, iccs_to_coords

SOFT_WARN_PLIES = 8
# Old schema name still appears in some model habits; accept at dispatch only
# (not in TOOL_DEFINITIONS) so search does not die as Unknown tool.
PREVIEW_ALIASES = frozenset({"preview_move"})


def is_submit_ok(result: str) -> bool:
    return any(line.startswith("OK:") for line in str(result).splitlines())


class ObserveSession:
    """Live board stays read-only; preview walks a working copy."""

    def __init__(self, live: Board):
        self.live = live
        self.work = live.copy()
        self.path: list[str] = []

    def header(self) -> str:
        n = len(self.path)
        to_move = "红" if self.work.turn == "w" else "黑"
        lines = [f"live | work +{n} | to_move={to_move}", *self._path_event_lines()]
        if n >= SOFT_WARN_PLIES:
            lines.append(f"path 已 {n} 半步，仍须对 live 提交")
        return "\n".join(lines)

    def _path_event_lines(self) -> list[str]:
        """Readable event line for the current preview path (not full game history)."""
        n = len(self.path)
        if n == 0:
            return ["path: -"]
        history = list(getattr(self.work, "move_history", None) or [])
        slice_ = history[-n:] if len(history) >= n else []
        out = ["path:"]
        if len(slice_) != n:
            for i, move in enumerate(self.path, start=1):
                out.append(f"  {i}. {move}")
            return out
        for i, rec in enumerate(slice_, start=1):
            move = (rec.get("move") or "").strip().lower()
            move_zh = (rec.get("move_zh") or "").strip()
            piece = rec.get("piece") or ""
            color = "红" if isinstance(piece, str) and piece.isupper() else "黑"
            zh_part = f" {move_zh}" if move_zh else ""
            out.append(f"  {i}. {color} {move}{zh_part}")
        return out

    def wrap(self, body: str) -> str:
        return f"{self.header()}\n{body}"

    def reset(self) -> str:
        self.work = self.live.copy()
        self.path = []
        return self.wrap("工作盘已回到 live。")

    def preview(self, iccs: str) -> str:
        move = (iccs or "").strip().lower()
        if not _ICCS.fullmatch(move):
            return self.wrap(f"Invalid move format: '{iccs}'. Must be ICCS (e.g., h2e2).")
        if not self.work.is_valid_move(move):
            return self.wrap(self._illegal_preview_body(move))
        _cf, _rf, ct, rt = iccs_to_coords(move)
        captured = self.work.get_piece(ct, rt)
        zh = self.work.to_chinese_move(move)
        count = occurrence_after(self.work, move)
        kind = ""
        if count == 2:
            nxt = self.work.copy()
            nxt.make_move(move)
            kind = _cycle_kind_zh(self.work, nxt)
        self.work.make_move(move)
        self.path.append(move)
        eat = "无"
        if captured:
            eat = f"{_color_zh(captured)}{PIECE_NAMES_ZH.get(captured, captured)}"
        check = "是" if self.work._is_in_check(self.work.turn) else "否"
        lines = [f"preview {move} {zh}", f"吃：{eat} | 将：{check}"]
        if count == 2:
            lines.append(f"循环：此局面将第2次出现（{kind}）。再走回即禁。")
        return self.wrap("\n".join(lines))

    def _illegal_preview_body(self, move: str) -> str:
        if self.work.turn != self.live.turn:
            return f"Illegal move: '{move}'. 工作盘已轮到对方。"
        return f"Illegal move: '{move}'."

    def submit_live(self, iccs: str) -> str:
        move = (iccs or "").strip().lower()
        if not _ICCS.fullmatch(move):
            return self.wrap(
                f"Invalid move format: '{iccs}'. Must be 4 characters in ICCS format (e.g., h2e2)."
            )
        live_ok = self.live.is_valid_move(move)
        work_ok = self.work.is_valid_move(move)
        if work_ok and not live_ok:
            return self.wrap(
                f"Rejected: '{move}' is legal on the working board, not on live. "
                f"make_move 只能提交真实盘己方下一手。"
            )
        if not live_ok:
            try:
                fen_after = self.live._trial_fen_after(move)
            except Exception:
                fen_after = ""
            if fen_after and is_threefold_after(self.live.position_keys, fen_after):
                return self.wrap(
                    f"Illegal move: '{move}' (third occurrence of this position is forbidden) "
                    f"(live, not work). 此着走后局面将第三次出现，禁止。不要再交同一着。"
                )
            return self.wrap(
                f"Illegal move: '{move}' (live, not work). "
                f"不是当前真实盘合法着，不要再交同一着。"
            )
        return self.wrap(f"OK: Move {move} is valid and will be played.")

    def dispatch(self, tool_name: str, args: dict | None = None) -> str:
        payload = args if isinstance(args, dict) else {}
        name = tool_name or ""
        if name in PREVIEW_ALIASES:
            name = "preview"
        if name == "preview":
            return self.preview(payload.get("move", ""))
        if name == "preview_reset":
            return self.reset()
        if name == "get_legal_moves":
            return self.wrap(
                legal_moves_annotated(
                    self.work,
                    squares=payload.get("squares"),
                    pieces=payload.get("pieces"),
                )
            )
        if name == "get_threats":
            return self.wrap(threats(self.work, self.work.turn))
        if name == "get_board":
            return self.wrap(
                f"{work_turn_line(self.work)}\nFEN：{self.work.to_fen()}\n{board_diagram(self.work)}"
            )
        if name == "make_move":
            return self.submit_live(payload.get("move", ""))
        if name == "preview_undo":
            return self.wrap(
                "此工具已移除。换分支请 preview_reset；"
                "若需保留已走前缀，reset 后重走公共前缀再 preview。"
            )
        if name == "get_recent_moves":
            return self.wrap(
                "此工具已移除。对手上一手已在系统快照；"
                "预演路径见工具头 path: 事件线。"
            )
        return self.wrap(f"Unknown tool: {tool_name}")
