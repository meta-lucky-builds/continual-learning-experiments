# T-S5 (ts5_plan.md). Resumable: skips any run whose .npz exists.
cd /mnt/project-files/notes/cifar
export CL_K=100 CL_NTASK=20 CIFAR_DIR=data100 RESV=20 LR=0.03
DLR=$(python3 -c "import json;print(json.load(open('results/sleep_tune/choice.json'))['dlr'])")
DEP=$(python3 -c "import json;print(json.load(open('results/sleep_tune/choice.json'))['dep'])")
MM=$(python3 -c "import json;print(json.load(open('results/ts5/budget.json'))['M_match'])")
O=results/ts5
for s in 0 1 2 3 4; do
  [ -f $O/erp_$s.npz ] || echo "MEM=2000 python3 cifar2.py run erp $s $O mem=2000 promote=1 > $O/erp_$s.log 2> $O/erp_$s.err"
  [ -f $O/er_mmatch_$s.npz ] || echo "MEM=2000 python3 cifar2.py run er_mmatch $s $O mem=$MM > $O/er_mmatch_$s.log 2> $O/er_mmatch_$s.err"
  [ -f $O/slp_$s.npz ] || echo "MEM=2000 python3 sleep.py run slp $s $O sleep=1 rank=16 dream=buf dlr=$DLR dep=$DEP > $O/slp_$s.log 2> $O/slp_$s.err"
  [ -f $O/rc_$s.npz ] || echo "MEM=2000 python3 cifar.py e1 rc $s $O > $O/rc_$s.log 2> $O/rc_$s.err"
done > results/jobs_ts5
wc -l results/jobs_ts5
xargs -P 4 -I{} bash -c "{}" < results/jobs_ts5; echo DONE
