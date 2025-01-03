import json
import cv2
import numpy as np
import os
from torch.utils.data import Dataset
from PIL import Image
import cv2
import sys
sys.path.append("..")
sys.path.append(".")
from utils import * 
from .base_unet import BaseDataset_unet
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



class YoutubeVISDataset_unet(BaseDataset_unet):
    def __init__(self, image_dir, anno, meta, tokenizer, sub_size, transforms, ytbvis_subset_size=40000, args = None):
        self.image_root = image_dir
        self.anno_root = anno 
        self.meta_file = meta

        video_dirs = []
        with open(self.meta_file) as f:
            records = json.load(f)
            records = records["videos"]
            for video_id in records:
                video_dirs.append(video_id)

        self.records = records
        self.data = video_dirs
        self.size = (512,512)
        self.clip_size = (224,224)
        self.dynamic = 1
        self.sub_size=sub_size
        self.tokenizer = tokenizer
        self.ytbvis_subset_size=ytbvis_subset_size
        self.transforms = transforms
        self.args = args

    def __len__(self):
        if self.ytbvis_subset_size is not None:
            return self.ytbvis_subset_size
        return 40000

    
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
        inputs = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        return inputs.input_ids
    
    def get_sample(self, idx):
        video_id = list(self.records.keys())[idx]
        objects_id = np.random.choice( list(self.records[video_id]["objects"].keys()) )
        frames = self.records[video_id]["objects"][objects_id]["frames"]
        object_cate = self.records[video_id]["objects"][objects_id]['category']
        
        prompt_template = random.choice(prompt_templates)
        target_prompt = prompt_template.format(object_cate)
        subject_prompt = f"a photo of a {object_cate}."
        
        # Sampling frames
        min_interval = len(frames)  // 8
        start_frame_index = np.random.randint(low=0, high=len(frames) - min_interval)
        end_frame_index = start_frame_index + np.random.randint(min_interval,  len(frames) - start_frame_index )
        end_frame_index = min(end_frame_index, len(frames) - 1)

        # Get image path
        sub_image_name = frames[start_frame_index]
        tar_image_name = frames[end_frame_index]
        sub_image_path = os.path.join(self.image_root, video_id, sub_image_name) + '.jpg'
        tar_image_path = os.path.join(self.image_root, video_id, tar_image_name) + '.jpg'
        sub_mask_path = sub_image_path.replace('JPEGImages','Annotations').replace('.jpg', '.png')
        tar_mask_path = tar_image_path.replace('JPEGImages','Annotations').replace('.jpg', '.png')

        sub_image = cv2.imread(sub_image_path)
        sub_image = cv2.cvtColor(sub_image, cv2.COLOR_BGR2RGB)

        tar_image = cv2.imread(tar_image_path)
        tar_image = cv2.cvtColor(tar_image, cv2.COLOR_BGR2RGB)

        sub_mask = Image.open(sub_mask_path ).convert('P')
        sub_mask= np.array(sub_mask)
        sub_mask = sub_mask == int(objects_id)
        
        tar_mask = Image.open(tar_mask_path).convert('P')
        tar_mask= np.array(tar_mask)
        tar_mask = tar_mask == int(objects_id)
        item_with_collage = self.process_pairs(sub_image, sub_mask, tar_image, tar_mask, sub_size=self.sub_size, transforms=self.transforms, args = self.args)
        sampled_time_steps = self.sample_timestep()
        item_with_collage['time_steps'] = sampled_time_steps
        item_with_collage['input_ids'] = self.tokenize_text(target_prompt)
        item_with_collage['subject_input_ids'] = self.tokenize_text(subject_prompt)
        item_with_collage['dataset_name'] = "youtubeVIS"
        item_with_collage['target_prompt'] = target_prompt
        item_with_collage['subject_prompt'] = subject_prompt
        
        return item_with_collage