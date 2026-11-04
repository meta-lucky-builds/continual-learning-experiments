cd /mnt/project-files/notes/cifar
for c in "0.01 10" "0.01 30" "0.03 10" "0.03 30"; do set -- $c; n=tune_$1_$2
  [ -f results/sleep_tune/${n}_100.npz ] && continue
  echo "RESV=20 LR=0.03 python3 sleep.py run $n 100 results/sleep_tune sleep=1 rank=16 dream=buf dlr=$1 dep=$2 > results/sleep_tune/${n}_100.log 2> results/sleep_tune/${n}_100.err"
done > results/jobs_sleep_tune
xargs -P 4 -I{} bash -c "{}" < results/jobs_sleep_tune; echo DONE
