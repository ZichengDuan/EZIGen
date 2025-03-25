# accelerate launch --config_file configs/training_config.yaml 
# accelerate launch infer_sdxl.py \
python infer_sdxl.py \
    --config configs/infer_config_sdxl.yaml \
    --guidance_scale 7\
    --seed 283 \
    --split_ratio 0.7 \
    --infer_steps 20 \
    --sim_threshold 0.99 \
    --target_prompt "A portrait of a white man wearing a rainbow scarf. master piece, 4k, ultra fine" \
    --subject_prompt "a man" \
    --subject_img_path "/mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/subjects/000000023_white.png" \
    --output_root "outputs/" \
    --num_interations 7 \
    # --init_img_path "/mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/source_images_with_masks/space_dog_chowchow_mask.png"