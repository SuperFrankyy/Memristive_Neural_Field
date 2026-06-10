#!/bin/bash

# 需要训练的场景列表
# scenes=("chair" "drums" "ficus" "hotdog" "materials" "mic" "ship" "lego")
scenes=("ship")

# 训练参数（可根据需要修改）
netwidth=128
netdepth=8
netwidth_fine=128
netdepth_fine=8
rank=32
rank_fine=32
pruningratio=0.9
basedir="./logs/train"
suffix="train"

for scene in "${scenes[@]}"
do
    config="configs/${scene}.txt"

    # 修改配置文件参数
    sed -i "s|basedir = .*|basedir = ${basedir}|g" $config
    sed -i "s|netwidth = .*|netwidth = ${netwidth}|g" $config
    sed -i "s|netdepth = .*|netdepth = ${netdepth}|g" $config
    sed -i "s|netwidth_fine = .*|netwidth_fine = ${netwidth_fine}|g" $config
    sed -i "s|netdepth_fine = .*|netdepth_fine = ${netdepth_fine}|g" $config
    sed -i "s|rank = .*|rank = ${rank}|g" $config
    sed -i "s|rank_fine = .*|rank_fine = ${rank_fine}|g" $config
    sed -i "s|pruningratio = .*|pruningratio = ${pruningratio}|g" $config
    sed -i "s|expname = .*|expname = ${scene}_w${netwidth}*${netdepth}_r${rank}_pruning${pruningratio}_${suffix}|g" $config

    # 启动训练
    python run_nerf.py --config $config
done