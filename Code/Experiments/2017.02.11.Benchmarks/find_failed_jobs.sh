#!/bin/sh

for i in $(seq 1 121); do
    ERRORLINES=$(cat slurm-12187645_$i.out | grep "Error using parpool" | wc -l)
    if [ $ERRORLINES -gt 0 ]; then
	echo $i >> failed_jobs.txt
    fi
done
