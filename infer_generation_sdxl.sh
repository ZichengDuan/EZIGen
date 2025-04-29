# accelerate launch --config_file configs/training_config.yaml 
# accelerate launch infer_sdxl.py \
python infer_sdxl.py \
    --config configs/infer_config_sdxl.yaml \
    --guidance_scale 7\
    --seed 284 \
    --split_ratio 0.5 \
    --infer_steps 50     \
    --sim_threshold 0.99 \
    --target_prompt "A corgi dog walking in the rain, wearing a top black hat" \
    --subject_prompt "a corgi dog" \
    --subject_img_path "example_images/subjects/dog.png" \
    --output_root "outputs/" \
    --num_interations 3 \
    # --init_img_path "/mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/source_images_with_masks/space_dog_chowchow_mask.png"