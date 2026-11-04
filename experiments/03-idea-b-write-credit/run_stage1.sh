#!/bin/bash
# Stage 1 job queue: 4 parallel single-thread workers.
cd "$(dirname "$0")"
{
  for s in 0 1 2 3 4 5 6 7 8 9; do
    echo "A1_dense $s"; echo "A2_dense_curric $s"; echo "A3_dense_local $s"
  done
  for s in 0 1 2; do echo "A0_naive $s"; done
} | xargs -P 4 -L 1 bash -c 'python3 hay.py --arm $0 --seed $1 --threads 1 >> logs/stage1.log 2>&1'
echo STAGE1_FINISHED >> logs/stage1.log
