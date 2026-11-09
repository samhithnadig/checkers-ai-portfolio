"""Evaluate a checkpoint against random and alpha-beta baselines."""
import argparse, torch
from .engine import GameState, RED
from .network import PolicyValueNet
from .mcts import MCTS
from .agents import RandomAgent, AlphaBetaAgent

class NeuralAgent:
    def __init__(self,model,simulations,device): self.search=MCTS(model,simulations,device=device)
    def select_move(self,state): return self.search.select(state,temperature=0)[0]

def play_game(red,black,max_plies=300):
    state=GameState.initial()
    for _ in range(max_plies):
        if state.winner is not None or state.draw: break
        agent=red if state.turn==RED else black
        move=agent.select_move(state)
        if move is None: break
        state.apply(move)
    if state.winner is None and not state.draw: state.draw=True
    return state.winner,state.draw

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--checkpoint",required=True)
    p.add_argument("--games",type=int,default=50)
    p.add_argument("--simulations",type=int,default=64)
    p.add_argument("--depth",type=int,default=3)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    a=p.parse_args()
    ckpt=torch.load(a.checkpoint,map_location=a.device)
    model=PolicyValueNet().to(a.device); model.load_state_dict(ckpt["model"]); model.eval()
    neural=NeuralAgent(model,a.simulations,a.device)
    for name,baseline in [("random",RandomAgent()),("alphabeta",AlphaBetaAgent(a.depth))]:
        wins=draws=losses=0
        for i in range(a.games):
            # Alternate colors to reduce first-player bias.
            if i%2==0: winner,isdraw=play_game(neural,baseline)
            else: winner,isdraw=play_game(baseline,neural)
            if isdraw: draws+=1
            elif (winner==RED) == (i%2==0): wins+=1
            else: losses+=1
        print(f"vs {name}: W-D-L {wins}-{draws}-{losses} over {a.games} games")
if __name__=="__main__": main()
