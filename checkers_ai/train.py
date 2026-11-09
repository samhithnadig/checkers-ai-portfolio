"""Self-play training for the policy/value network."""
import argparse, random
from collections import deque
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
from .engine import GameState
from .features import board_features, move_features
from .network import PolicyValueNet
from .mcts import MCTS

def self_play_game(model, simulations, device, max_plies=240):
    state=GameState.initial()
    samples=[]
    for ply in range(max_plies):
        if state.winner is not None or state.draw: break
        search=MCTS(model,simulations=simulations,device=device)
        counts,root=search.run(state)
        if not counts: break
        moves=list(counts)
        visits=np.asarray([counts[m] for m in moves],dtype=np.float32)
        policy=visits/visits.sum() if visits.sum() else np.ones_like(visits)/len(visits)
        bf=board_features(state.board,state.turn)
        mfs=np.stack([move_features(m,state.board,state.turn) for m in moves])
        samples.append((bf,mfs,policy,state.turn))
        if ply < 20:
            weights=np.power(visits,1.0/1.0)
            probs=weights/weights.sum() if weights.sum() else np.ones_like(weights)/len(weights)
            move=moves[int(np.random.choice(len(moves),p=probs))]
        else:
            move=moves[int(np.argmax(visits))]
        state.apply(move)
    if state.winner is None and not state.draw: state.draw=True
    result=[]
    for bf,mfs,policy,player in samples:
        target=0.0 if state.draw else (1.0 if state.winner==player else -1.0)
        result.append((bf,mfs,policy,target))
    return result, state.winner, state.draw, state.ply

def train(args):
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device=args.device
    model=PolicyValueNet().to(device)
    optimizer=torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=1e-4)
    replay=deque(maxlen=args.buffer)
    for game in range(1,args.games+1):
        model.eval()
        data,winner,draw,plies=self_play_game(model,args.simulations,device,args.max_plies)
        replay.extend(data)
        if len(replay)>=args.batch_size:
            model.train()
            for _ in range(args.epochs):
                batch=random.sample(list(replay),min(args.batch_size,len(replay)))
                losses=[]
                for bf,mfs,policy,target in batch:
                    b=torch.tensor(bf[None],device=device)
                    m=torch.tensor(mfs[None],device=device)
                    target_policy=torch.tensor(policy[None],device=device)
                    target_value=torch.tensor([target],dtype=torch.float32,device=device)
                    logits,value=model(b,m)
                    policy_loss=-(target_policy*F.log_softmax(logits,dim=-1)).sum(dim=-1).mean()
                    value_loss=F.mse_loss(value,target_value)
                    losses.append(policy_loss+value_loss)
                optimizer.zero_grad()
                torch.stack(losses).mean().backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
        print(f"game {game}/{args.games}: plies={plies}, winner={winner}, draw={draw}, replay={len(replay)}")
        if game % args.save_every==0 or game==args.games:
            Path(args.output).parent.mkdir(parents=True,exist_ok=True)
            torch.save({"model":model.state_dict(),"games":game,"config":vars(args)},args.output)
            print(f"saved {args.output}")
    return model

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--games",type=int,default=100)
    p.add_argument("--simulations",type=int,default=64)
    p.add_argument("--epochs",type=int,default=2)
    p.add_argument("--batch-size",type=int,default=32)
    p.add_argument("--buffer",type=int,default=5000)
    p.add_argument("--max-plies",type=int,default=240)
    p.add_argument("--lr",type=float,default=1e-3)
    p.add_argument("--device",default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed",type=int,default=7)
    p.add_argument("--save-every",type=int,default=10)
    p.add_argument("--output",default="checkpoints/policy_value.pt")
    train(p.parse_args())
if __name__=="__main__": main()
