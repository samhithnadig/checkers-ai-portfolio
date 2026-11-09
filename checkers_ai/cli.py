import argparse
from .engine import GameState, perft
def main():
    p=argparse.ArgumentParser(); sub=p.add_subparsers(dest="cmd",required=True)
    q=sub.add_parser("perft"); q.add_argument("--depth",type=int,default=5)
    a=p.parse_args()
    if a.cmd=="perft":
        s=GameState.initial()
        print(perft(s.board,s.turn,a.depth))
if __name__=="__main__": main()
