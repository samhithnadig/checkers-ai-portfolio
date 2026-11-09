"""Run a game between baseline agents and print the board after each ply."""
import argparse
from .engine import GameState, RED, render
from .agents import RandomAgent, AlphaBetaAgent

def make_agent(name,depth):
    if name=="random": return RandomAgent()
    if name=="alphabeta": return AlphaBetaAgent(depth)
    raise ValueError("Supported agents: random, alphabeta")

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--red",choices=["random","alphabeta"],default="random")
    p.add_argument("--black",choices=["random","alphabeta"],default="random")
    p.add_argument("--depth",type=int,default=3)
    p.add_argument("--max-plies",type=int,default=300)
    a=p.parse_args()
    state=GameState.initial()
    print(render(state.board))
    for _ in range(a.max_plies):
        if state.winner is not None or state.draw: break
        agent=make_agent(a.red if state.turn==RED else a.black,a.depth)
        move=agent.select_move(state)
        if move is None: break
        print(f"\n{'RED' if state.turn==RED else 'BLACK'}: {move}")
        state.apply(move)
        print(render(state.board))
    print("Result:", "draw" if state.draw else ("red wins" if state.winner==RED else "black wins" if state.winner==-RED else "unfinished"))

if __name__=="__main__": main()
