#!/bin/bash

# accelerate launch --config_file configs/training_config.yaml 
# accelerate launch infer_flux.py \
python infer_flux.py \
    --config configs/flux/infer_config_flux.yaml \
    --guidance_scale 3.5 \
    --seed 42 \
    --split_ratio 0.9 \
    --infer_steps 40 \
    --sim_threshold 0.99 \
    --target_prompt "A robotic horse dog standing in front of the pyramid." \
    --subject_prompts "a robotic horse" \
    --subject_img_paths "example_images/subjects/cyber_horse.png" \
    --output_root "outputs/" \
    --num_interations 5 \
    # --init_img_path "/path/to/initial/image.png"