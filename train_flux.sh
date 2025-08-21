#!/bin/zsh
export NCCL_DEBUG=INFO
export NO_ALBUMENTATIONS_UPDATE=1

# === 手动定义节点名和每个节点的 GPU 编号 ===
NODES=("g081" "g067" "g092")  # 顺序决定 node_rank
NODES=("g061" "g055")  # 顺序决定 node_rank
# NODES=("g061")
CUDA_VISIBLE_DEVICES_LIST=("0,1,2,3" "0,1,2,3" "0,1")  # 一一对应
CUDA_VISIBLE_DEVICES_LIST=("0,1,2,3" "0,1,2,3")  # 一一对应
# CUDA_VISIBLE_DEVICES_LIST=("0,1,2,3")  # 一一对应
# CUDA_VISIBLE_DEVICES_LIST=("0")  # 一一对应

# === 自动识别当前节点名 ===
HOSTNAME=$(hostname)


# 默认 config 文件
DEFAULT_CONFIG="configs/flux/flux_vanilla_ft.yaml"

# 解析命令行参数
while [ $# -gt 0 ]; do
  case "$1" in
    --config)
      CONFIG_PATH="$2"
      shift 2  # 跳过 --config 和参数值
      ;;
    *)
      echo "Unknown option: $1"
      exit 1
      ;;
  esac
done

# 如果没有提供 --config 参数，则使用默认值
CONFIG_PATH=${CONFIG_PATH:-$DEFAULT_CONFIG}

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

# 运行训练
# accelerate launch --config_file /datastore/zha414/dzc/Projects/CogVideo/finetune/accelerate_config_4gpu_bf16.yaml train_flux.py --config "$CONFIG_PATH"
torchrun \
    --nproc_per_node=$GPUS_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --rdzv_id=5235 \
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:29500 \
    train_flux.py --config "$CONFIG_PATH"
# torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --master_port=2614 train_sdxl.py --config "$CONFIG_PATH"