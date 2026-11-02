#!/bin/bash
# one job: name \t cfg ; seeds 0-4
name="$1"; cfg="$2"
NT=1 python3 -I /mnt/project-files/notes/marathon/mar2.py run "$cfg" 0,1,2,3,4 > /mnt/project-files/notes/marathon/results/$name.jsonl 2> /mnt/project-files/notes/marathon/results/$name.err
