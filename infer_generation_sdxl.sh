
# accelerate launch --config_file configs/training_config.yaml 
python infer_sdxl.py \
    --config configs/infer_config_sdxl.yaml \
    --guidance_scale 10\
    --seed 123 \
    --split_ratio 0.5 \
    --infer_steps 50 \
    --sim_threshold 0.99 \
    --target_prompt "a white and black border collie on a tropical beach." \
    --subject_prompt "a dog" \
    --subject_img_path "example_images/subjects/dog8_04.png" \
    --output_root "outputs/" \
    --num_interations 6
    # --init_img_path "example_images/source_images_with_masks/monster.png" \
    