# Resumable: skips any run whose .npz exists. Hyperparameters from results/sleep_tune/choice.json (seed 100).
cd /mnt/project-files/notes/cifar
DLR=$(python3 -c "import json;print(json.load(open('results/sleep_tune/choice.json'))['dlr'])")
DEP=$(python3 -c "import json;print(json.load(open('results/sleep_tune/choice.json'))['dep'])")
{ for s in 0 1 2 3 4; do echo "erw $s width=64"; done
  for s in 0 1 2 3 4; do
    echo "slp_r16_buf $s sleep=1 rank=16 dream=buf dlr=$DLR dep=$DEP"; echo "slp_r4_buf $s sleep=1 rank=4 dream=buf dlr=$DLR dep=$DEP"
    echo "slp_full_buf $s sleep=1 rank=0 dream=buf dlr=$DLR dep=$DEP"; echo "slp_full_mix $s sleep=1 rank=0 dream=mix dlr=$DLR dep=$DEP"
    echo "slp_full_noise $s sleep=1 rank=0 dream=noise dlr=$DLR dep=$DEP"; done; } | while read name s args; do
  [ -f results/sleep/${name}_$s.npz ] && continue
  echo "RESV=20 LR=0.03 python3 sleep.py run $name $s results/sleep $args > results/sleep/${name}_$s.log 2> results/sleep/${name}_$s.err"
done > results/jobs_sleep_main
wc -l results/jobs_sleep_main
xargs -P 4 -I{} bash -c "{}" < results/jobs_sleep_main; echo DONE
