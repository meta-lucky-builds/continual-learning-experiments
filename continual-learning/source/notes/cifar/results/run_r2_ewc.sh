#!/bin/bash
cd /mnt/project-files/notes/cifar
until grep -q DONE results/run_r2.log 2>/dev/null; do sleep 30; done
RESV=20 NT=4 python3 pick_ewc.py > results/r2/pick_ewc.log 2>&1 || exit 1
LAM=$(python3 -c "import json;print(json.load(open('results/r2/ewc_lambda.json'))['lam'])")
export NT=1
for s in 0 1 2 3 4; do echo "RESV=20 LR=0.03 python3 cifar2.py run ewc $s results/r2 ewc=$LAM > results/r2/ewc_$s.log 2> results/r2/ewc_$s.err"; done | xargs -P 4 -I{} bash -c "{}"
echo DONE
