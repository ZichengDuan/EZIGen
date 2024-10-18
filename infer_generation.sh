
python infer.py \
    --config configs/infer_config.yaml \
    --guidance_scale 7.5\
    --seed 3154 \
    --split_ratio 0.4 \
    --infer_steps 50 \
    --sim_threshold 0.99 \
    --tar_prompt "a dog in police outfit" \
    --sub_prompt "a dog" \
    --sub_img_path "example_images/subjects/dog6.png" \
    --output_root "outputs/" \
    # --num_interations 6