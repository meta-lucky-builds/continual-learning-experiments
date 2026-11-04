#!/bin/bash
# resumable: skips any job whose output npz already exists (container restarts kill background jobs)
cd /mnt/project-files/notes/cifar
export NT=1
{ for s in 0 1 2 3 4; do
  echo "m2c $s m2c=1"; echo "er_e3 $s epochs=3"; echo "erp $s promote=1"; echo "er_m680 $s mem=680"; echo "er_m1200 $s mem=1200"; echo "ewc $s ewc=10.0"; echo "rcw $s -"
done; } | sort -s -k1,1 -t' ' --key=1,1 | awk '{print ($1=="m2c"?0:1), $0}' | sort -s -n -k1,1 | cut -d' ' -f2- | while read name s arg; do
  [ -f results/r2/${name}_$s.npz ] && continue
  if [ "$name" = rcw ]; then echo "RESV=20 python3 cifar.py e1 rcw $s results/r2 > results/r2/rcw_$s.log 2> results/r2/rcw_$s.err"
  else echo "RESV=20 LR=0.03 python3 cifar2.py run $name $s results/r2 $arg > results/r2/${name}_$s.log 2> results/r2/${name}_$s.err"; fi
done > results/jobs_r2_resume
wc -l results/jobs_r2_resume
xargs -P 4 -I{} bash -c "{}" < results/jobs_r2_resume
echo DONE
