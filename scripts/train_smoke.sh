#!/usr/bin/env bash
set -euo pipefail
python -m checkers_ai.train --games 2 --simulations 4 --epochs 1 --batch-size 4 --max-plies 80 --save-every 2 --device cpu --output checkpoints/smoke.pt
