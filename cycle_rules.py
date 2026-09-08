"""Pikafish-style cycle rules: count positions, classify check/chase, ban threefold."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


def position_key(board) -> str:
    return board.to_fen()


def count_after(keys: Iterable[str], fen_after: str) -> int:
    return sum(1 for key in keys if key == fen_after) + 1


def is_threefold_after(keys: Iterable[str], fen_after: str) -> bool:
    return count_after(keys, fen_after) >= 3


def occurrence_after(board, move_iccs: str) -> int:
    keys = getattr(board, "position_keys", None) or [board.to_fen()]
    return count_after(keys, board._trial_fen_after(move_iccs))


def current_occurrence(board) -> int:
    fen = board.to_fen()
    keys = getattr(board, "position_keys", None) or [fen]
    return sum(1 for key in keys if key == fen)


def filter_cycle_moves(board, moves: list[str]) -> list[str]:
    keys = getattr(board, "position_keys", None) or [board.to_fen()]
    kept: list[str] = []
    for move in moves:
        fen_after = board._trial_fen_after(move)
        if not is_threefold_after(keys, fen_after):
            kept.append(move)
    return kept


@dataclass(frozen=True)
class CycleClass:
    red: str
    black: str
    red_level: int
    black_level: int
    note: str


_LEVEL = {"check": 2, "chase": 1, "idle": 0}


def classify_cycle(board) -> CycleClass | None:
    """Classify the latest closed cycle on board.position_keys, if any."""
    keys = list(getattr(board, "position_keys", None) or [])
    history = list(getattr(board, "move_history", None) or [])
    if len(keys) < 3 or len(history) < 2:
        return None
    current = keys[-1]
    try:
        first = keys.index(current)
    except ValueError:
        return None
    end_ply = len(keys) - 1
    if first == end_ply or end_ply - first < 2 or end_ply > len(history):
        return None

    from xiangqi import Board

    pieces = _snapshot_pieces(Board(keys[first]))
    plies = []
    for ply in range(first, end_ply):
        rec = history[ply]
        before = Board(rec["fen_before"])
        after = Board(keys[ply + 1])
        mover = "w" if rec["piece"] in "KABNRCP" else "b"
        opponent = "b" if mover == "w" else "w"
        gives_check = after._is_in_check(after.turn)
        chased_after = set() if gives_check else _chase_victim_ids(after, mover, pieces)
        chased_before_opp = _chase_victim_ids(before, opponent, pieces)
        pieces = _apply_move_ids(pieces, rec)
        chased_after_opp = _chase_victim_ids(after, opponent, pieces)
        plies.append(
            {
                "mover": mover,
                "check": gives_check,
                "chased": chased_after,
                "escaped": chased_before_opp - chased_after_opp,
            }
        )

    red_kind = _side_kind(plies, "w")
    black_kind = _side_kind(plies, "b")
    return CycleClass(
        red=red_kind,
        black=black_kind,
        red_level=_LEVEL[red_kind],
        black_level=_LEVEL[black_kind],
        note=_note_for(red_kind, black_kind),
    )


def _snapshot_pieces(board) -> dict[int, tuple[str, int, int]]:
    pieces: dict[int, tuple[str, int, int]] = {}
    n = 0
    for row in range(10):
        for col in range(9):
            piece = board.get_piece(col, row)
            if piece is None:
                continue
            pieces[n] = (piece, col, row)
            n += 1
    return pieces


def _apply_move_ids(pieces: dict[int, tuple[str, int, int]], rec: dict) -> dict[int, tuple[str, int, int]]:
    fc, fr = rec["from"]
    tc, tr = rec["to"]
    moved = {
        i: val
        for i, val in pieces.items()
        if not (val[1] == tc and val[2] == tr)
    }
    for i, (piece, col, row) in list(moved.items()):
        if (col, row) == (fc, fr):
            moved[i] = (piece, tc, tr)
            break
    return moved


def _id_at(pieces: dict[int, tuple[str, int, int]], col: int, row: int) -> int | None:
    for i, (_piece, c, r) in pieces.items():
        if (c, r) == (col, row):
            return i
    return None


def _side_kind(plies: list[dict], side: str) -> str:
    own = [p for p in plies if p["mover"] == side]
    opp = [p for p in plies if p["mover"] != side]
    if not own:
        return "idle"
    if any(p["check"] for p in plies):
        if all(p["check"] for p in own):
            return "check"
        return "idle"
    if not opp:
        return "idle"
    victims = set(own[0]["chased"])
    for ply in own[1:]:
        victims &= ply["chased"]
    if not victims:
        return "idle"
    for victim in victims:
        if all(victim in ply["escaped"] for ply in opp):
            return "chase"
    return "idle"


def _chase_victim_ids(board, attacker_side: str, pieces: dict[int, tuple[str, int, int]]) -> set[int]:
    victims: set[int] = set()
    saved = board.turn
    board.turn = attacker_side
    try:
        for move in board._legal_moves_ignoring_cycle():
            cf, rf, ct, rt = _iccs(move)
            attacker = board.get_piece(cf, rf)
            victim = board.get_piece(ct, rt)
            if victim is None:
                continue
            if not _is_chase_attack(board, attacker, victim, cf, rf, ct, rt):
                continue
            vid = _id_at(pieces, ct, rt)
            if vid is not None:
                victims.add(vid)
    finally:
        board.turn = saved
    return victims


def _is_chase_attack(board, attacker, victim, cf, rf, ct, rt) -> bool:
    if attacker is None or victim is None:
        return False
    if victim.lower() == "k":
        return False
    if attacker.lower() in {"k", "p"}:
        return False
    if victim.lower() == "p" and not _pawn_crossed(victim, rt):
        return False
    if attacker.lower() == victim.lower() and _square_attacked_by_piece(
        board, cf, rf, ct, rt, victim
    ):
        return False
    if _attacker_hangs_after_capture(board, attacker, victim, cf, rf, ct, rt):
        if _hang_exception(attacker, victim):
            return True
        return False
    return True


def _hang_exception(attacker, victim) -> bool:
    a = attacker.lower()
    v = victim.lower()
    if a in {"n", "c"} and v == "r":
        return True
    if a in {"a", "b"} and v in {"n", "c", "r"}:
        return True
    return False


def _attacker_hangs_after_capture(board, attacker, victim, cf, rf, ct, rt) -> bool:
    from xiangqi import Board

    nxt = Board(board.to_fen())
    nxt._grid[rf][cf] = None
    nxt._grid[rt][ct] = attacker
    nxt.turn = "b" if board.piece_color(attacker) == "w" else "w"
    for move in nxt._legal_moves_ignoring_cycle():
        _cf, _rf, tcol, trow = _iccs(move)
        if (tcol, trow) == (ct, rt):
            return True
    return False


def _square_attacked_by_piece(board, target_col, target_row, from_col, from_row, piece) -> bool:
    if board.get_piece(from_col, from_row) != piece:
        return False
    return (target_col, target_row) in board._attack_squares_for(from_col, from_row)


def _pawn_crossed(pawn: str, row: int) -> bool:
    if pawn == "P":
        return row >= 5
    return row <= 4


def _iccs(move: str):
    from xiangqi import iccs_to_coords

    return iccs_to_coords(move)


def _note_for(red_kind: str, black_kind: str) -> str:
    parts = []
    label = {"check": "长将", "chase": "长捉", "idle": "允许循环"}
    if red_kind != "idle":
        parts.append(f"红{label[red_kind]}")
    if black_kind != "idle":
        parts.append(f"黑{label[black_kind]}")
    if not parts:
        return "双方允许循环"
    return "，".join(parts)
