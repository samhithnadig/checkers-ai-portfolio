"""Numerical board and legal-move features."""
import numpy as np

def board_features(board, turn):
    # Four planes: own men/kings, opponent men/kings; perspective is side-to-move.
    x = np.zeros((4,8,8), dtype=np.float32)
    for r in range(8):
        for c in range(8):
            p = int(board[r,c])
            if p == 0: continue
            own = p * turn > 0
            plane = (0 if own else 2) + (1 if abs(p)==2 else 0)
            x[plane,r,c] = 1.0
    return x

def move_features(move, board, turn):
    sr,sc=move.start; er,ec=move.end
    piece=int(board[sr,sc])
    return np.asarray([
        sr/7, sc/7, er/7, ec/7,
        (er-sr)/7, (ec-sc)/7,
        len(move.captures)/7,
        float(abs(piece)==2),
        float(abs(piece)==1 and ((turn==1 and er==0) or (turn==-1 and er==7))),
        float(move.is_capture),
    ], dtype=np.float32)
