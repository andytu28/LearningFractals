#!/bin/bash


GPU=$1

for IDX in `seq 0 9`;
do
    for SAMPLE_IDX in `seq 0 9`;
    do
        bash run_reconstruct.sh $GPU kmnist $IDX $SAMPLE_IDX
    done
done
