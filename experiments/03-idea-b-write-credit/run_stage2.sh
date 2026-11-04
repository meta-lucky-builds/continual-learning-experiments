#!/bin/bash
# Stage 2 (pre-registered, conditional on A3 > A1 at p < 0.05): locality control A4, seeds 0-9.
cd "$(dirname "$0")"
for s in 0 1 2 3 4 5 6 7 8 9; do echo "A4_dense_end $s"; done \
 | xargs -P 4 -L 1 bash -c 'python3 hay.py --arm $0 --seed $1 --threads 1 >> logs/stage2.log 2>&1'
echo STAGE2_FINISHED >> logs/stage2.log
