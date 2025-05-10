# accelerate launch --config_file configs/training_config.yaml 
# accelerate launch infer_sdxl.py \
python infer_sdxl.py \
    --config configs/infer_config_sdxl.yaml \
    --guidance_scale 10\
    --seed 285 \
    --split_ratio 0.8 \
    --infer_steps 50     \
    --sim_threshold 0.99 \
    --target_prompt "A robotic horse dog standing in front of the pyramid." \
    --subject_prompt "a robotic horse" \
    --subject_img_path "example_images/subjects/cyber_horse.png" \
    --output_root "outputs/" \
    --num_interations 5 \
    # --init_img_path "/mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/source_images_with_masks/space_dog_chowchow_mask.png"