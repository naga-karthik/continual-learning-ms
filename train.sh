#!/bin/bash

seeds='5 9 46 171 1113'
for seed in $seeds; do
    CUDA_VISIBLE_DEVICES=2 python main_pl_replay_random.py -nspv 4 -dep 3 -initf 32 -ps 64 -me 150 -bs 4 -cve 10 -se $seed -lr 1e-4
done 

echo "---------------------------------------------------------"
echo "Training and testing done"
echo "---------------------------------------------------------"
