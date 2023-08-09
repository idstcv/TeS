#!/bin/bash

dataname='cifar100';
num_classes=100;
log=${dataname};
python run.py \
    -j 32 \
    --lr 0.00316 \
    --wd 0.00001 \
    --pretrained your/checkpoint/path \
    --path-label-train-file ./cifar100/trainval.txt \
    --path-label-test-file ./cifar100/test.txt \
    --batch-size 256 --multiprocessing-distributed --world-size 1 --rank 0 \
    --num-classes ${num_classes} --nesterov True --dist-url 'tcp://127.0.0.1:'$((RANDOM%10000+10000)) \
    --log ${log} --cos-lr \
    --mean-class-acc 0 \
    --lambda_v 0.1 --lambda_t 0.3 --tau 0.03 \
    your/cifar100/data/path | tee ./${log}.log