# infer_editing.sh
python infer_sdxl.py \
    --config configs/infer_config_sdxl.yaml \
    --guidance_scale 8\
    --seed 1234 \
    --split_ratio 0.8 \
    --infer_steps 30 \
    --sim_threshold 0.99 \
    --target_prompt "a chow chow dog in astronaut outfit" \
    --subject_prompt "a chow chow dog" \
    --subject_img_path "/mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/subjects/chowchow.png" \
    --output_root "outputs/" \
    --source_image_path /mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/source_images_with_masks/space_dog_chowchow.png \
    --do_editing \
    --num_interations 8 \
    --foreground_mask_path /mnt/sh_nas/duanzicheng.dzc/Projects/EZIGen/example_images/source_images_with_masks/space_dog_chowchow_mask.png