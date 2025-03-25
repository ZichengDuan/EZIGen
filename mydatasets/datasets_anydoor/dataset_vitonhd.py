import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
from .data_utils import * 
from .base import BaseDataset_unet
import albumentations as A
import random


prompt_templates = [
    "a fine image of {}.",
    "a {} center at the middle of the image.",
    "a picture of {}.",
    "a clear shot of {}.",
    "an image showing a {}.",
    "a simple depiction of {}.",
    "a plain view of {}.",
    "a basic image of {}.",
    "a straightforward representation of {}.",
    "an ordinary photo of {}.",
    "a visual of {}.",
    "an everyday look at {}.",
    "a usual sight of {}.",
    "a typical image of {}.",
    "a common depiction of {}.",
    "a straightforward photo of {}.",
    "a classic view of {}.",
    "a generic image of {}.",
    "a standard depiction of {}.",
    "a direct view of {}.",
    "an uncomplicated image of {}.",
    "a true representation of {}.",
    "an unembellished image of {}.",
    "a neat picture of {}.",
    "a regular view of {}.",
    "a plain photo of {}.",
    "a bare image of {}.",
    "an unadorned view of {}.",
    "a straightforward portrayal of {}.",
    "a direct depiction of {}."
]

class VitonHDDataset_unet(BaseDataset_unet):
    def __init__(self, image_dir, sub_size, transforms, vitonhd_subset_size=40000, args = None, tokenizer_one=None, tokenizer_two=None):
        self.image_root = image_dir
        self.data = os.listdir(self.image_root)
        self.size = (1024,1024)
        self.clip_size = (224,224)
        self.dynamic = 2
        self.tokenizer_one=tokenizer_one
        self.tokenizer_two=tokenizer_two
        self.vitonhd_subset_size = vitonhd_subset_size
        self.args = args
        self.transforms=transforms
        self.sub_size=sub_size



    def __len__(self):
        return self.vitonhd_subset_size

    def check_region_size(self, image, yyxx, ratio, mode = 'max'):
        pass_flag = True
        H,W = image.shape[0], image.shape[1]
        H,W = H * ratio, W * ratio
        y1,y2,x1,x2 = yyxx
        h,w = y2-y1,x2-x1
        if mode == 'max':
            if h > H and w > W:
                pass_flag = False
        elif mode == 'min':
            if h < H and w < W:
                pass_flag = False
        return pass_flag

    
    def tokenize_text(self, text):
        inputs_one = self.tokenizer_one(
            text, max_length=self.tokenizer_one.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        one_ids = inputs_one.input_ids
        
        if self.tokenizer_two:
            inputs_two = self.tokenizer_two(
                text, max_length=self.tokenizer_two.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
            )
            two_ids = inputs_two.input_ids
        else:
            two_ids = None
        
        return one_ids, two_ids


    def get_sample(self, idx):
        idx = idx % len(self.data)

        ref_image_path = os.path.join(self.image_root, self.data[idx])
        tar_image_path = ref_image_path.replace('/cloth/', '/image/')
        ref_mask_path = ref_image_path.replace('/cloth/','/cloth-mask/')
        tar_mask_path = ref_image_path.replace('/cloth/', '/image-parse-v3/').replace('.jpg','.png')

        prompt_template = random.choice(prompt_templates)
        target_prompt = prompt_template.format("picture of a model wearing cloth")
        subject_prompt = f"a photo of a cloth."

        # Read Image and Mask
        ref_image = cv2.imread(ref_image_path)
        ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        ref_mask = (cv2.imread(ref_mask_path) > 128).astype(np.uint8)[:,:,0]

        tar_mask = Image.open(tar_mask_path ).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == 5

        item_with_collage = self.process_pairs(ref_image, ref_mask, tar_image, tar_mask, sub_size=self.sub_size, transforms=self.transforms, args = self.args, max_ratio = 1.0)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['input_ids'], item_with_collage['input_ids_two'] = self.tokenize_text(target_prompt)
        item_with_collage['subject_input_ids'], item_with_collage['subject_input_ids_two'] = self.tokenize_text(subject_prompt)
        item_with_collage['dataset_name'] = "vitonHD"
        item_with_collage['target_prompt'] = target_prompt
        item_with_collage['subject_prompt'] = subject_prompt
        item_with_collage['padding_num'] = self.args.num_sub_img - 1


        return item_with_collage
