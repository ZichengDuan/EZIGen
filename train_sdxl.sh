#!/bin/zsh
export NCCL_DEBUG=INFO
# 默认 config 文件
DEFAULT_CONFIG="configs/train_config_sdxl.yaml"

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

# 获取 CUDA_VISIBLE_DEVICES 的值（如果未设置，则默认为所有 GPU）
CUDA_DEVICES=${CUDA_VISIBLE_DEVICES:-$(nvidia-smi --query-gpu=index --format=csv,noheader | paste -sd "," -)}
# CUDA_DEVICES=1

# 计算可用 GPU 的数量
NUM_GPUS=$(echo $CUDA_DEVICES | awk -F',' '{print NF}')

# 自动设置配置文件名称
CONFIG_FILE="m1_r${NUM_GPUS}_bf16.yaml"

echo "Detected $NUM_GPUS GPU(s), using config: $CONFIG_FILE"
echo "Training with additional config: $CONFIG_PATH"

# 运行训练
accelerate launch --config_file accelerate_configs/$CONFIG_FILE train_sdxl.py --config "$CONFIG_PATH"
# python train_sdxl.py --config "$CONFIG_PATH"
# torchrun --nproc_per_node=$NUM_GPUS --nnodes=1 --master_port=2614 train_sdxl.py --config "$CONFIG_PATH"