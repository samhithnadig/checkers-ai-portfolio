"""Classical baselines."""
import math, random
from .engine import RED, BLACK, Move, legal_moves

class RandomAgent:
    def select_move(self, state):
        moves = state.legal_moves()
        return random.choice(moves) if moves else None

def evaluate(board, perspective):
    score = 0.0
    for r in range(8):
        for c in range(8):
            p = int(board[r,c])
            if not p: continue
            value = 1.0 if abs(p)==1 else 1.8
            advancement = (7-r)/7 if p > 0 else r/7
            value += 0.12 * advancement if abs(p)==1 else 0
            score += value if p*perspective > 0 else -value
    return score

class AlphaBetaAgent:
    def __init__(self, depth=4): self.depth = depth
    def select_move(self, state):
        moves = state.legal_moves()
        if not moves: return None
        maximizing = state.turn
        best_move, best_score = moves[0], -math.inf
        for move in moves:
            child = state.clone(); child.apply(move)
            score = self._search(child, self.depth-1, -math.inf, math.inf, maximizing)
            if score > best_score: best_score, best_move = score, move
        return best_move
    def _search(self, state, depth, alpha, beta, perspective):
        if state.winner is not None: return 1000 if state.winner == perspective else -1000
        if state.draw: return 0
        moves = state.legal_moves()
        if not moves: return -1000
        if depth <= 0: return evaluate(state.board, perspective)
        if state.turn == perspective:
            value = -math.inf
            for m in moves:
                child=state.clone(); child.apply(m)
                value=max(value,self._search(child,depth-1,alpha,beta,perspective))
                alpha=max(alpha,value)
                if beta<=alpha: break
            return value
        value=math.inf
        for m in moves:
            child=state.clone(); child.apply(m)
            value=min(value,self._search(child,depth-1,alpha,beta,perspective))
            beta=min(beta,value)
            if beta<=alpha: break
        return value
