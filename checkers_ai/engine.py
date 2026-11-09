"""English draughts rules engine. Board values: +1 red man, +2 red king, -1 black man, -2 black king."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable
import numpy as np

EMPTY = 0
RED, BLACK = 1, -1
BOARD_SIZE = 8
DARK_SQUARES = tuple((r, c) for r in range(8) for c in range(8) if (r + c) % 2 == 1)

@dataclass(frozen=True, order=True)
class Move:
    path: tuple[tuple[int, int], ...]
    captures: tuple[tuple[int, int], ...] = ()

    @property
    def start(self): return self.path[0]
    @property
    def end(self): return self.path[-1]
    @property
    def is_capture(self): return bool(self.captures)

    def __str__(self):
        sep = "x" if self.is_capture else "-"
        return sep.join(f"{r},{c}" for r, c in self.path)

@dataclass
class GameState:
    board: np.ndarray
    turn: int = RED
    no_progress: int = 0
    ply: int = 0
    winner: int | None = None
    draw: bool = False
    history: dict[tuple, int] | None = None

    def __post_init__(self):
        # The current position is already occurrence one, including for
        # custom positions constructed directly by callers.
        if self.history is None:
            self.history = {}
        self.history.setdefault(self.position_key(), 1)

    @classmethod
    def initial(cls):
        b = np.zeros((8, 8), dtype=np.int8)
        for r, c in DARK_SQUARES:
            if r < 3: b[r, c] = BLACK
            elif r > 4: b[r, c] = RED
        return cls(b, RED, history={})

    def clone(self):
        return GameState(self.board.copy(), self.turn, self.no_progress, self.ply,
                         self.winner, self.draw, dict(self.history or {}))

    def position_key(self):
        return (self.board.tobytes(), self.turn)

    def legal_moves(self):
        return legal_moves(self.board, self.turn)

    def apply(self, move: Move, draw_limit: int = 80):
        if self.winner is not None or self.draw:
            raise ValueError("Game is already over")
        legal = self.legal_moves()
        if move not in legal:
            raise ValueError(f"Illegal move: {move}")
        old_piece = int(self.board[move.start])
        self.board[move.start] = EMPTY
        for sq in move.captures:
            self.board[sq] = EMPTY
        end_r, end_c = move.end
        piece = old_piece
        promoted = (abs(piece) == 1 and ((piece == RED and end_r == 0) or (piece == BLACK and end_r == 7)))
        if promoted: piece = 2 * self.turn
        self.board[end_r, end_c] = piece
        self.ply += 1
        self.no_progress = 0 if move.is_capture or promoted else self.no_progress + 1
        self.turn = -self.turn
        key = self.position_key()
        self.history[key] = self.history.get(key, 0) + 1
        red_count = int(np.sum(self.board > 0))
        black_count = int(np.sum(self.board < 0))
        if red_count == 0: self.winner = BLACK
        elif black_count == 0: self.winner = RED
        elif not self.legal_moves(): self.winner = -self.turn
        elif self.no_progress >= draw_limit or self.history[key] >= 3: self.draw = True
        return self

def _directions(piece: int):
    if abs(piece) == 2: return ((-1,-1),(-1,1),(1,-1),(1,1))
    dr = -1 if piece > 0 else 1
    return ((dr,-1),(dr,1))

def _inside(r,c): return 0 <= r < 8 and 0 <= c < 8

def _capture_sequences(board: np.ndarray, start: tuple[int,int], piece: int):
    results = []
    def rec(b, pos, path, captures, current_piece):
        found = False
        r,c = pos
        for dr,dc in _directions(current_piece):
            mr,mc = r+dr,c+dc
            lr,lc = r+2*dr,c+2*dc
            if not (_inside(lr,lc) and _inside(mr,mc)): continue
            if b[mr,mc] * current_piece < 0 and b[lr,lc] == EMPTY:
                found = True
                nb = b.copy()
                nb[r,c] = EMPTY
                nb[mr,mc] = EMPTY
                promoted = abs(current_piece) == 1 and ((current_piece == RED and lr == 0) or (current_piece == BLACK and lr == 7))
                next_piece = 2 * (1 if current_piece > 0 else -1) if promoted else current_piece
                nb[lr,lc] = next_piece
                npath = path + ((lr,lc),)
                ncaps = captures + ((mr,mc),)
                # In English draughts, a man reaching the king row during a capture ends its turn.
                if promoted:
                    results.append(Move(npath, ncaps))
                else:
                    rec(nb, (lr,lc), npath, ncaps, next_piece)
        if not found and captures:
            results.append(Move(path, captures))
    rec(board.copy(), start, (start,), (), piece)
    return results

def legal_moves(board: np.ndarray, turn: int) -> list[Move]:
    captures = []
    quiet = []
    for r,c in DARK_SQUARES:
        piece = int(board[r,c])
        if piece * turn <= 0: continue
        captures.extend(_capture_sequences(board, (r,c), piece))
        for dr,dc in _directions(piece):
            nr,nc = r+dr,c+dc
            if _inside(nr,nc) and board[nr,nc] == EMPTY:
                quiet.append(Move(((r,c),(nr,nc))))
    return sorted(captures if captures else quiet)

def perft(board: np.ndarray, turn: int, depth: int) -> int:
    if depth == 0: return 1
    moves = legal_moves(board, turn)
    total = 0
    for m in moves:
        b = board.copy()
        p = int(b[m.start]); b[m.start] = EMPTY
        for sq in m.captures: b[sq] = EMPTY
        er,ec = m.end
        if abs(p) == 1 and ((p == RED and er == 0) or (p == BLACK and er == 7)): p = 2*turn
        b[er,ec] = p
        total += perft(b, -turn, depth-1)
    return total

def render(board: np.ndarray) -> str:
    symbols = {0: ".", 1: "r", 2: "R", -1: "b", -2: "B"}
    lines = ["  0 1 2 3 4 5 6 7"]
    for r in range(8): lines.append(str(r)+" "+" ".join(symbols[int(x)] for x in board[r]))
    return "\n".join(lines)
