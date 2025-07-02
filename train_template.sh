# python train/train_distill_align_feat.py --config_path configs/distill_align_feat/wan14B_2_wan1_3B.yaml
# accelerate launch --config_file configs/accelerate/4gpu_bf16.yaml train/train_distill_align_feat.py --config_path configs/distill_align_feat/wan14B_2_wan1_3B.yaml

#!/bin/bash

# === 手动定义节点名和每个节点的 GPU 编号 ===
NODES=("g081" "g067" "g092")  # 顺序决定 node_rank
NODES=("g093" "g054")  # 顺序决定 node_rank
# NODES=("g092")
CUDA_VISIBLE_DEVICES_LIST=("0,1,2,3" "0,1,2,3" "0,1")  # 一一对应
CUDA_VISIBLE_DEVICES_LIST=("0,1,2,3" "0,1,2,3")  # 一一对应
# CUDA_VISIBLE_DEVICES_LIST=("0")  # 一一对应

# === 自动识别当前节点名 ===
HOSTNAME=$(hostname)

# === 查找当前节点对应的 node_rank 和 CUDA_VISIBLE_DEVICES ===
for i in "${!NODES[@]}"; do
    if [[ "$HOSTNAME" == "${NODES[$i]}" ]]; then
        NODE_RANK=$i
        export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES_LIST[$i]}
        break
    fi
done

# === 验证是否成功识别 ===
if [[ -z "$NODE_RANK" ]]; then
    echo "Error: This hostname ($HOSTNAME) is not in the defined NODES list."
    exit 1
fi

MASTER_ADDR=${NODES[0]}  # 默认第一个节点为 master
NNODES=${#NODES[@]}
GPUS_PER_NODE=$(echo $CUDA_VISIBLE_DEVICES | awk -F',' '{print NF}')

echo "Node: $HOSTNAME"
echo "NODE_RANK=$NODE_RANK"
echo "MASTER_ADDR=$MASTER_ADDR"
echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

# === 启动 accelerate launch ===
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    causvid/train_distillation.py \
    --config_path  configs/wan_causal_dmd_1_frame.yaml