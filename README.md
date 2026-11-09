# Checkers AI: From Rules Engine to Self-Play Learning

A portfolio project that builds an English draughts (American checkers) engine, compares classical agents, and trains a small policy/value neural network through self-play.

## What it demonstrates
- A tested game engine with mandatory captures, multi-jumps, kings, and promotion rules.
- Random and alpha-beta search baselines.
- A PyTorch policy/value model that scores **legal candidate moves** (rather than relying on a fragile fixed move-index mapping).
- PUCT Monte Carlo Tree Search and self-play training.
- Reproducible evaluation: trained agent vs random and alpha-beta baselines.

This is a focused project: it trains an agent to play checkers, not a general-purpose model that can do literally anything. That narrower scope is much more credible in a portfolio.

## Requirements
- Python 3.10+
- NumPy
- PyTorch (optional; needed only for neural training/inference)
- pytest (tests)

## Setup (Linux/macOS)
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q
# Only if you want neural-network training locally:
pip install -r requirements-training.txt
```

If your laptop has limited storage or no GPU, install the CPU build of PyTorch from the official PyTorch selector. You can run the engine and alpha-beta baseline without a GPU. For training, install `requirements-training.txt`; alternatively use Kaggle or Colab and install it there. The engine, tests, and alpha-beta baseline do not require PyTorch.

## Quick start
```bash
# Play a random-vs-random game
python -m checkers_ai.play --red random --black alphabeta --depth 3

# Check engine perft counts
python -m checkers_ai.cli perft --depth 6

# Train (small smoke run)
python -m checkers_ai.train --games 20 --simulations 32 --epochs 2 --device cpu

# Evaluate a checkpoint
python -m checkers_ai.evaluate --checkpoint checkpoints/policy_value.pt --games 50 --simulations 64
```

Training is compute-intensive. Start with the smoke-run settings, then increase games and simulations on a GPU.

## Rules and conventions
- English draughts on an 8x8 board; only dark squares are playable.
- `+1` = red man, `+2` = red king, `-1` = black man, `-2` = black king.
- Red starts at the bottom and moves toward decreasing row numbers. Black starts at the top and moves toward increasing row numbers.
- Captures are mandatory. A multi-jump is one complete move. A man that reaches its promotion row during a capture is crowned and its turn ends (English draughts rule).
- A draw is declared after the configurable no-progress limit or a repeated position.

## Project layout
```text
checkers-ai-portfolio/
├── checkers_ai/
│   ├── engine.py       # Board, move generation, game state, perft
│   ├── agents.py       # Random and alpha-beta agents
│   ├── features.py     # Board and candidate-move features
│   ├── network.py      # PyTorch policy/value network
│   ├── mcts.py         # PUCT search
│   ├── train.py        # Self-play + replay buffer + training
│   ├── evaluate.py     # Tournament evaluation
│   ├── play.py         # CLI game runner
│   └── cli.py
├── tests/
└── scripts/
```

## Honest portfolio claims
Once you have run the experiments, report actual results: games trained, hardware, wall-clock time, win/draw/loss rates, and baseline comparison. Do not claim the model is strong until it beats a meaningful baseline over a sufficiently large match set.
