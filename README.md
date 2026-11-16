> [!WARNING]
> 🚧 **Work in progress.** The engine and alpha-beta baseline are done. Self-play training and evaluation are still underway, and results will be added soon.

# Checkers AI: From Rules Engine to Self-Play Learning

An English draughts (American checkers) engine built from scratch, with classical search baselines and a small policy/value neural network trained through self-play.

## Status

- ✅ Game engine with perft-verified move generation
- ✅ Random and alpha-beta baselines
- ⏳ Self-play training
- ⏳ Evaluation and results

## What it demonstrates

- A tested game engine with mandatory captures, multi-jumps, kings, and promotion rules.
- Random and alpha-beta search baselines.
- A PyTorch policy/value model that scores **legal candidate moves**, rather than relying on a fragile fixed move-index mapping.
- PUCT Monte Carlo Tree Search and self-play training.
- Reproducible evaluation: the trained agent vs. random and alpha-beta baselines.

## Requirements

- Python 3.10+
- NumPy
- pytest (for tests)
- PyTorch (optional, only needed for neural network training and inference)

## Setup (Linux/macOS)

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pytest -q

# Only if you want to train the neural network locally:
pip install -r requirements-training.txt
```

The engine, tests, and alpha-beta baseline run without PyTorch or a GPU. If you have limited storage or no GPU, install the CPU build of PyTorch from the official PyTorch selector, or run training on Kaggle or Colab instead.

## Quick start

```bash
# Play random vs. alpha-beta (depth 3)
python -m checkers_ai.play --red random --black alphabeta --depth 3

# Check engine perft counts
python -m checkers_ai.cli perft --depth 6

# Train (small smoke run)
python -m checkers_ai.train --games 20 --simulations 32 --epochs 2 --device cpu

# Evaluate a checkpoint
python -m checkers_ai.evaluate --checkpoint checkpoints/policy_value.pt --games 50 --simulations 64
```

Training is compute-intensive. Start with the smoke-run settings, then increase games and simulations on a GPU.

## Engine verification

Perft counts every legal move sequence from the starting position up to a given depth, which is a standard way to check that move generation is correct. Reference counts for English draughts, where a multi-jump counts as one move:

| Depth | Positions |
|------:|----------:|
| 1 | 7 |
| 2 | 49 |
| 3 | 302 |
| 4 | 1,469 |
| 5 | 7,361 |
| 6 | 36,768 |

## Rules and conventions

- English draughts on an 8×8 board. Only dark squares are playable.
- `+1` = red man, `+2` = red king, `-1` = black man, `-2` = black king.
- Red starts at the bottom and moves toward decreasing row numbers. Black starts at the top and moves toward increasing row numbers.
- Captures are mandatory. A multi-jump counts as one complete move.
- A man that reaches its promotion row during a capture is crowned, and its turn ends.
- A game is drawn after a configurable no-progress limit or a repeated position.

## Project layout

```text
checkers-ai-portfolio/
├── checkers_ai/
│   ├── engine.py      # Board, move generation, game state, perft
│   ├── agents.py      # Random and alpha-beta agents
│   ├── features.py    # Board and candidate-move features
│   ├── network.py     # PyTorch policy/value network
│   ├── mcts.py        # PUCT search
│   ├── train.py       # Self-play, replay buffer, training loop
│   ├── evaluate.py    # Tournament evaluation
│   ├── play.py        # CLI game runner
│   └── cli.py         # CLI utilities (perft)
├── tests/
└── scripts/
```
