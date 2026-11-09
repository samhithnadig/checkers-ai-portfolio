"""PUCT MCTS over legal moves, using a policy/value network."""
import math, numpy as np, torch
from .features import board_features, move_features
from .engine import RED

class Node:
    def __init__(self, state, prior=1.0):
        self.state=state; self.prior=prior; self.visits=0; self.value_sum=0.0
        self.children={}
    @property
    def q(self): return self.value_sum/self.visits if self.visits else 0.0

class MCTS:
    def __init__(self, model, simulations=64, c_puct=1.4, device="cpu"):
        self.model=model; self.simulations=simulations; self.c_puct=c_puct; self.device=device

    @staticmethod
    def _selection_score(child, parent_visits, c_puct):
        # Child Q is stored from the child's side-to-move perspective.
        # The parent therefore sees the negated value.
        exploration = c_puct * child.prior * math.sqrt(max(1, parent_visits)) / (1 + child.visits)
        return -child.q + exploration

    @torch.no_grad()
    def _evaluate(self, state):
        moves=state.legal_moves()
        if not moves: return moves, np.zeros(0,dtype=np.float32), -1.0
        bf=torch.tensor(board_features(state.board,state.turn)[None],device=self.device)
        mf=torch.tensor(np.stack([move_features(m,state.board,state.turn) for m in moves])[None],device=self.device)
        logits,value=self.model(bf,mf)
        probs=torch.softmax(logits[0],dim=0).cpu().numpy()
        return moves,probs,float(value.item())

    def run(self, state):
        root=Node(state.clone())
        moves,priors,_=self._evaluate(root.state)
        if not moves: return {}, root
        root.children={m:Node(self._next_state(root.state,m),float(p)) for m,p in zip(moves,priors)}
        for _ in range(self.simulations):
            node=root; path=[root]
            while node.children:
                total=max(1,node.visits)
                move,child=max(node.children.items(),
                    key=lambda kv: self._selection_score(kv[1], total, self.c_puct))
                node=child; path.append(node)
                if node.state.winner is not None or node.state.draw: break
            if node.state.winner is not None:
                value=1.0 if node.state.winner==node.state.turn else -1.0
            elif node.state.draw: value=0.0
            else:
                moves,priors,value=self._evaluate(node.state)
                node.children={m:Node(self._next_state(node.state,m),float(p)) for m,p in zip(moves,priors)}
            # Value is from leaf side-to-move perspective; alternate sign while backing up.
            for n in reversed(path):
                n.visits += 1; n.value_sum += value
                value = -value
        visit_counts={m:ch.visits for m,ch in root.children.items()}
        return visit_counts,root

    @staticmethod
    def _next_state(state, move):
        child=state.clone(); child.apply(move); return child

    def select(self, state, temperature=1.0):
        counts,root=self.run(state)
        if not counts: return None, counts
        moves=list(counts); visits=np.asarray([counts[m] for m in moves],dtype=np.float64)
        if temperature <= 1e-6: idx=int(np.argmax(visits))
        else:
            weights=np.power(visits,1.0/temperature)
            if weights.sum()==0: weights=np.ones_like(weights)
            probs=weights/weights.sum(); idx=int(np.random.choice(len(moves),p=probs))
        return moves[idx],counts
