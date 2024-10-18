python infer.py \
    --config configs/infer_config.yaml \
    --guidance_scale 7.5\
    --seed 3154 \
    --split_ratio 0.4 \
    --infer_steps 50 \
    --sim_threshold 0.99 \
    --tar_prompt "a woman" \
    --sub_prompt "a woman" \
    --sub_img_path "example_images/subjects/lifeifei.png" \
    --output_root "outputs/" \
    --foreground_mask_path example_images/source_images_with_masks/woman_mask.png \
    --source_image_path example_images/source_images_with_masks/woman.png \
    --do_editing
    # --num_interations 6