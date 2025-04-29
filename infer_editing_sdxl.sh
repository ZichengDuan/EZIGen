# infer_editing.sh
python infer_sdxl.py \
    --config configs/infer_config_sdxl.yaml \
    --guidance_scale 7\
    --seed 283 \
    --split_ratio 0.6 \
    --infer_steps 50 \
    --sim_threshold 0.99 \
    --target_prompt "a robotic horse riding by a cowboy" \
    --subject_prompt "a robotic horse" \
    --subject_img_path "example_images/subjects/cyber_horse.png" \
    --output_root "outputs/" \
    --source_image_path example_images/source_images_with_masks/cowboy_horse.png \
    --do_editing \
    --num_interations 5 \
    --foreground_mask_path example_images/source_images_with_masks/cowboy_horse_mask.png