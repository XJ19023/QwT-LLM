#!/usr/bin/env bash

CONFIG=$1
CHECKPOINT=$2
GPUS=$3

random_port=$((RANDOM % 65535 + 1024))

# 检查端口是否被占用
while [[ "$(netstat -tuln | grep -w "$random_port")" != "" ]]; do
    random_port=$((RANDOM % 65535 + 1024))
done

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=$GPUS --master_port=$random_port \
    $(dirname "$0")/test.py $CONFIG $CHECKPOINT --launcher pytorch ${@:4}
