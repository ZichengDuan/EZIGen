#!/bin/zsh

# 检查可用的 GPU 数量
num_gpus=$(nvidia-smi -L | wc -l)

if [ "$num_gpus" -eq 0 ]; then
    echo "No GPUs found."
    exit 1
fi

echo "Detected $num_gpus GPUs."
out_dir="/mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/output_batched"


# 运行任务
for ((i = 1; i < $num_gpus + 1; i++)); do
    gpu_index=$((i - 1))  # 使用当前循环的索引作为GPU索引

    # 运行脚本并将其置于后台
    CUDA_VISIBLE_DEVICES=$gpu_index python infer_sdxl_batched.py \
        --config configs/infer_config_sdxl.yaml \
        --guidance_scale 10 \
        --split_ratio 0.5 \
        --infer_steps 20 \
        --sim_threshold 0.99 \
        --output_root $out_dir \
        --num_interations 8 \
        --batch_idx $i \
        --batch_total_num $num_gpus > $out_dir/$i.txt 2>&1 &

    echo "Started job $i on GPU $gpu_index"
done

wait  # 等待所有后台作业完成
