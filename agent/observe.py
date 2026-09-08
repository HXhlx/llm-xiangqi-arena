"""Rule-only observation for live LLM players. No engine, no scores."""

from __future__ import annotations

import re
from collections import OrderedDict
from typing import Optional

from cycle_rules import classify_cycle, current_occurrence, occurrence_after
from xiangqi import PIECE_NAMES_ZH, RED_PIECES, Board, iccs_to_coords

SCORE_FORBIDDEN = (
    "分数",
    "评分",
    "centipawn",
    "elo",
    "multipv",
    "bestmove",
    "最佳着",
    "最佳",
    "略优",
    "明显优势",
    "大优",
    "劣势",
    "推荐",
    "engine",
    "pikafish",
    "stockfish",
    "eval",
    "score",
    "advantage",
)

_ICCS = re.compile(r"[a-i][0-9][a-i][0-9]")


def board_diagram(board: Board) -> str:
    landing = _last_landing(board)
    lines = ["  a b c d e f g h i", "  +-+-+-+-+-+-+-+-+"]
    for row in range(9, -1, -1):
        parts = [f"{row} "]
        for col in range(9):
            piece = board.get_piece(col, row)
            ch = piece if piece else "."
            parts.append(f"{ch}*" if landing == (col, row) else f"{ch} ")
        parts.append(f" {row}")
        lines.append("".join(parts))
        if row == 5:
            lines.append("  = = = 楚河汉界 = = =")
    lines.append("  +-+-+-+-+-+-+-+-+")
    lines.append("  a b c d e f g h i")
    if landing:
        lines.append(f"上一手落点: {_sq(*landing)} *")
    return "\n".join(lines)


def file_legend(side: str) -> str:
    if side == "b":
        files = "黑一路(a)…黑九路(i)"
    else:
        files = "红九路(a)…红一路(i)"
    return (
        f"路：{files}。红方从右到左一路至九路，黑方从左到右一路至九路。"
        "九宫：红 d0–f2，黑 d7–f9。河界在第4–5行之间。"
    )


def work_turn_zh(board: Board) -> str:
    return "红" if board.turn == "w" else "黑"


def work_turn_line(board: Board) -> str:
    return f"以下为工作盘轮走方={work_turn_zh(board)}"


def threats(board: Board, side: str) -> str:
    bits: list[str] = []
    if board._is_in_check(side):
        bits.append("己方正在被将军")
    seen = current_occurrence(board)
    if seen >= 2:
        bits.append(f"此局面已出现 {seen} 次，走回同一局面的着已禁")
    captures = _legal_captures(board)
    if captures:
        bits.append(f"轮走方={work_turn_zh(board)} 可吃：" + "；".join(captures))
    for motif in _motifs(board):
        if motif not in bits:
            bits.append(motif)
    body = "无明显威胁事实。" if not bits else "；".join(bits) + "。"
    return f"{work_turn_line(board)}\n{body}"


def _legal_captures(board: Board) -> list[str]:
    """Legal captures this turn, labeled rooted / hanging."""
    out: list[str] = []
    seen: set[str] = set()
    for iccs in board.get_legal_moves():
        _cf, _rf, ct, rt = iccs_to_coords(iccs)
        victim = board.get_piece(ct, rt)
        if not victim:
            continue
        sq = _sq(ct, rt)
        name = f"{_color_zh(victim)}{PIECE_NAMES_ZH.get(victim, victim)}"
        key = f"{name}{sq}"
        if key in seen:
            continue
        seen.add(key)
        owner = board.piece_color(victim)
        rooted = bool(owner and board._is_attacked_by(ct, rt, owner))
        if rooted:
            root = "有根=被保护，仍可被吃"
        else:
            root = "无根=未被保护，当前可吃"
        out.append(f"{name} {sq}（{root}）")
    return out


def own_pieces(board: Board, side: str) -> str:
    """Own-side piece coordinates only — not legal moves."""
    items: list[str] = []
    for row in range(10):
        for col in range(9):
            piece = board.get_piece(col, row)
            if not piece or board.piece_color(piece) != side:
                continue
            items.append(f"{_color_zh(piece)}{PIECE_NAMES_ZH.get(piece, piece)} {_sq(col, row)}")
    return "，".join(items) if items else "无"


def _tokens(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        raw = [str(item) for item in value]
    else:
        raw = re.split(r"[,，\s]+", str(value))
    return [item.strip() for item in raw if item and str(item).strip()]


_SQUARE_RE = re.compile(r"[a-i][0-9]")
_PIECE_ALIASES = {
    "k": "K", "帅": "K", "将": "K", "king": "K", "general": "K",
    "a": "A", "仕": "A", "士": "A", "advisor": "A",
    "b": "B", "相": "B", "象": "B", "elephant": "B", "bishop": "B",
    "n": "N", "马": "N", "horse": "N", "knight": "N",
    "r": "R", "车": "R", "rook": "R", "chariot": "R",
    "c": "C", "炮": "C", "cannon": "C",
    "p": "P", "兵": "P", "卒": "P", "pawn": "P", "soldier": "P",
}


def normalize_squares(value: object) -> set[str]:
    out: set[str] = set()
    for tok in _tokens(value):
        text = tok.strip().lower()
        if _SQUARE_RE.fullmatch(text):
            out.add(text)
        elif _ICCS.fullmatch(text):
            out.add(text[:2])
    return out


def normalize_piece_kinds(value: object) -> set[str]:
    out: set[str] = set()
    for tok in _tokens(value):
        key = tok.strip().lower()
        if key in _PIECE_ALIASES:
            out.add(_PIECE_ALIASES[key])
        elif len(tok) == 1 and tok.upper() in "KABNRCP":
            out.add(tok.upper())
    return out


def filter_legal_moves(
    board: Board,
    squares: object = None,
    pieces: object = None,
) -> list[str]:
    moves = board.get_legal_moves()
    square_set = normalize_squares(squares)
    kinds = normalize_piece_kinds(pieces)
    if not square_set and not kinds:
        return moves
    out: list[str] = []
    for iccs in moves:
        if square_set and iccs[:2] not in square_set:
            continue
        if kinds:
            cf, rf, _ct, _rt = iccs_to_coords(iccs)
            piece = board.get_piece(cf, rf)
            if not piece or piece.upper() not in kinds:
                continue
        out.append(iccs)
    return out


def _annotate_move(board: Board, iccs: str) -> tuple[str, str]:
    cf, rf, ct, rt = iccs_to_coords(iccs)
    piece = board.get_piece(cf, rf)
    header = f"{_color_zh(piece)}{PIECE_NAMES_ZH.get(piece, piece or '')} {_sq(cf, rf)}"
    flags: list[str] = []
    if board.get_piece(ct, rt):
        flags.append("吃")
    nxt = board.copy()
    nxt.make_move(iccs)
    if nxt._is_in_check(nxt.turn):
        flags.append("将")
    if occurrence_after(board, iccs) == 2:
        flags.append("循环警告")
    zh = board.to_chinese_move(iccs)
    flag_s = "".join(f"[{f}]" for f in flags)
    line = f"  {zh} {iccs}" + (f" {flag_s}" if flag_s else "")
    return header, line


def legal_moves_annotated(
    board: Board,
    squares: object = None,
    pieces: object = None,
) -> str:
    prefix = work_turn_line(board)
    all_moves = board.get_legal_moves()
    if not all_moves:
        return f"{prefix}\n无合法着。"
    moves = filter_legal_moves(board, squares, pieces)
    if not moves:
        return f"{prefix}\n无匹配合法着。"
    groups: OrderedDict[str, list[str]] = OrderedDict()
    for iccs in moves:
        header, line = _annotate_move(board, iccs)
        groups.setdefault(header, []).append(line)
    body = "\n".join(
        f"{header}：" + "\n" + "\n".join(lines) for header, lines in groups.items()
    )
    return f"{prefix}\n{body}"


def _cycle_kind_zh(before: Board, after: Board) -> str:
    result = classify_cycle(after)
    if result is None:
        return "允许循环"
    mover = before.turn
    kind = result.red if mover == "w" else result.black
    return {"check": "长将", "chase": "长捉", "idle": "允许循环"}.get(kind, "允许循环")


def _last_landing(board: Board) -> Optional[tuple[int, int]]:
    if not board.move_history:
        return None
    last = board.move_history[-1]
    to_sq = last.get("to")
    if isinstance(to_sq, (tuple, list)) and len(to_sq) == 2:
        return int(to_sq[0]), int(to_sq[1])
    move = (last.get("move") or "").strip().lower()
    if len(move) >= 4 and _ICCS.fullmatch(move):
        return ord(move[2]) - ord("a"), int(move[3])
    return None


def _sq(col: int, row: int) -> str:
    return f"{chr(col + ord('a'))}{row}"


def _color_zh(piece: Optional[str]) -> str:
    if not piece:
        return ""
    return "红" if piece in RED_PIECES else "黑"


def _iter_pieces(board: Board):
    for row in range(10):
        for col in range(9):
            piece = board.get_piece(col, row)
            if piece:
                yield col, row, piece


def _aligned(c1: int, r1: int, c2: int, r2: int) -> bool:
    return c1 == c2 or r1 == r2


def _between(c1: int, r1: int, c2: int, r2: int) -> list[tuple[int, int]]:
    if c1 == c2:
        step = 1 if r2 > r1 else -1
        return [(c1, r) for r in range(r1 + step, r2, step)]
    if r1 == r2:
        step = 1 if c2 > c1 else -1
        return [(c, r1) for c in range(c1 + step, c2, step)]
    return []


def _pieces_between(board: Board, c1: int, r1: int, c2: int, r2: int) -> list[str]:
    found = []
    for c, r in _between(c1, r1, c2, r2):
        p = board.get_piece(c, r)
        if p:
            found.append(p)
    return found


def _motifs(board: Optional[Board]) -> list[str]:
    """Rule-only tactical motifs for the observe/threats surface (no engine)."""
    if not board:
        return []
    found: list[str] = []
    kings = {}
    for col, row, piece in _iter_pieces(board):
        if piece.upper() == "K":
            kings[board.piece_color(piece)] = (col, row)
    for col, row, piece in _iter_pieces(board):
        if piece.upper() == "C":
            enemy = "b" if board.piece_color(piece) == "w" else "w"
            kpos = kings.get(enemy)
            if kpos and _aligned(col, row, kpos[0], kpos[1]):
                mid = _pieces_between(board, col, row, kpos[0], kpos[1])
                if not mid:
                    found.append("空头炮")
            if (piece in RED_PIECES and row == 9) or (
                piece not in RED_PIECES and row == 0
            ):
                found.append("沉底炮")
        if piece.upper() == "N":
            if (piece in RED_PIECES and (col, row) == (4, 1)) or (
                piece not in RED_PIECES and (col, row) == (4, 8)
            ):
                found.append("窝心马")
    if kings.get("w"):
        door = board.get_piece(4, 1)
        if door and board.piece_color(door) == "w":
            found.append("将门被堵")
    if kings.get("b"):
        door = board.get_piece(4, 8)
        if door and board.piece_color(door) == "b":
            found.append("将门被堵")
    seen: set[str] = set()
    uniq = []
    for motif in found:
        if motif not in seen:
            seen.add(motif)
            uniq.append(motif)
    return uniq

