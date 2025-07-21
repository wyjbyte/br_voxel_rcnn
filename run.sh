#!/bin/bash     
export PYTHONUNBUFFERED=1
source activate voxelrcnn++
module unload gcc/12.4.0
# # 设置 PYTHONPATH
# export PYTHONPATH='/data/run01/sczc402/3dproject/voxelrcnn++:$PYTHONPATH'
# # 打印当前工作目录
# echo "Current working directory: $(pwd)"
# cd '/data/home/sczc402/run/3dproject/voxelrcnn++'


cd '/data/home/sczc402/run/3dproject/voxelrcnn++'
# 参数说明：分区名、任务名称、总GPU数量、其他参数
#python -m torch.distributed.run --nproc_per_node=2 tools/train.py  --launcher pytorch
CUDA_VISIBLE_DEVICES=0,1 python -m torch.distributed.run --nproc_per_node=2 tools/train.py  --launcher pytorch
#python tools/train.py
