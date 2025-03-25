import sys
sys.path.append("..")
sys.path.append(".")
import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
from transformers import CLIPTextModel, CLIPTokenizer, CLIPProcessor, CLIPModel, AutoImageProcessor, AutoModel, AutoTokenizer, CLIPTextModelWithProjection
import random
from utils.image_processors import random_horizontal_flip, random_rotation, random_scaling

class Subject200k_dataset_sdxl_jigsaw_sdxl(Dataset):
    def __init__(self, data_paths, transform=None, max_len=4, tokenizer=None, tokenizer_two=None, args=None, subset_size=None):
        self.data_paths = data_paths
        self.transform = transform
        # self.data_pairs = self._load_image_pairs()
        self.pair_indices = np.load("data/Subjects200K_collection3/unique_idxs.npy")
        self.args = args
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.tokenizer_two = tokenizer_two
        self.subset_size = subset_size if subset_size is not None else 50000
        self.jigsaw_size = 4  # 设定 batch size 为 4

    def _load_image_pairs(self):
        data_tuples = []
        idx = 0
        
        while True:
            if os.path.exists(os.path.join(self.data_paths, f"{idx}_subject.png")):
                data_tuples.append([
                    os.path.join(self.data_paths, f"{idx}_subject.png"),
                    os.path.join(self.data_paths, f"{idx}_target.png"),
                    os.path.join(self.data_paths, f"{idx}_prompts.txt")
                ])
                idx += 1
            else:
                break
        return data_tuples
    
    def tokenize_text(self, text):
        inputs_one = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
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
    
    def __len__(self):
        return self.subset_size  

    def __getitem__(self, idx):
        # batch_pairs = random.sample(self.data_pairs, self.jigsaw_size)
        
        pair_ids = random.sample(list(self.pair_indices), self.jigsaw_size)

        data_tuples = []
        for pair_id in pair_ids:
            data_tuples.append([
                        os.path.join(self.data_paths, f"{pair_id}_subject.png"),
                        os.path.join(self.data_paths, f"{pair_id}_target.png"),
                        os.path.join(self.data_paths, f"{pair_id}_prompts.txt")
                    ])

        left_images, right_images = [], []
        left_descriptions, right_descriptions = [], []
        
        for subject_image_path, target_image_path, prompts_file_path in data_tuples:
            left_image = Image.open(target_image_path).convert("RGB")  # 确保是 RGB
            right_image = Image.open(subject_image_path).convert("RGB")
            
            with open(prompts_file_path, "r") as prompts:
                lines = prompts.readlines()
                
            left_descriptions.append(lines[0].strip())
            right_descriptions.append(lines[1].strip())

            left_images.append(left_image)
            right_images.append(right_image)
        
        # **STEP 1: 先拼接 4 张 left_image（未 transform）**
        grid_size = 2  # 2x2 组合
        img_size = left_images[0].size[0] // 2  # 计算缩小后的尺寸
        resized_left_images = [img.resize((img_size, img_size), Image.BILINEAR) for img in left_images]
           
        # 创建 2x2 拼接目标图像
        target_image = Image.new("RGB", (img_size * 2, img_size * 2))
        target_image.paste(resized_left_images[0], (0, 0))
        target_image.paste(resized_left_images[1], (img_size, 0))
        target_image.paste(resized_left_images[2], (0, img_size))
        target_image.paste(resized_left_images[3], (img_size, img_size))

        if random.random() < 0.3 and self.args.with_rotation:
            target_image = random_rotation(target_image, max_angle=150)

        # **STEP 3: 统一对拼接后的 target_image 做 transform**
        target_image = self.transform(target_image)
        subject_images =[self.transform(img) for img in right_images]  # 重新 transform 右图
        subject_images = torch.stack(subject_images)
        
        # format descriptsions
        left_descripstion_combined = f"A four-panel collage image. Top left image: {left_descriptions[0]}, Top right image: {left_descriptions[1]}, bottom left image: {left_descriptions[2]}, bottom right image: {left_descriptions[3]}"
        
        if random.randint(1,100) / 100 < self.args.drop_text_ratio:
            left_descripstion_combined = ""

        one_ids, two_ids = self.tokenize_text(left_descripstion_combined)
        sub_one_ids, sub_two_ids = self.tokenize_text(right_descriptions)
        sample = {
            "target_prompt": left_descripstion_combined,
            "subject_prompt": right_descriptions,
            "target_image": target_image,  # 2x2 拼接后的左图
            "subject_images": subject_images,  # 4 张右图
            "input_ids": one_ids,
            "input_ids_two": two_ids,
            "subject_input_ids": sub_one_ids,
            "subject_input_ids_two": sub_two_ids,
            "dataset_name": "subject200k_jigsaw",
            "padding_num": 0,
        }
        
        return sample


class Subject200k_dataset_sdxl_jigsaw_sd21(Dataset):
    def __init__(self, data_paths, transform=None, max_len=4, tokenizer=None, args=None, subset_size=None):
        self.data_paths = data_paths
        self.transform = transform
        # self.data_pairs = self._load_image_pairs()
        self.pair_indices = np.load("data/Subjects200K_collection3/unique_idxs.npy")
        self.args = args
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.subset_size = subset_size if subset_size is not None else 50000
        self.jigsaw_size = 4  # 设定 batch size 为 4

    def _load_image_pairs(self):
        data_tuples = []
        idx = 0
        
        while True:
            if os.path.exists(os.path.join(self.data_paths, f"{idx}_subject.png")):
                data_tuples.append([
                    os.path.join(self.data_paths, f"{idx}_subject.png"),
                    os.path.join(self.data_paths, f"{idx}_target.png"),
                    os.path.join(self.data_paths, f"{idx}_prompts.txt")
                ])
                idx += 1
            else:
                break
        return data_tuples
    
    def tokenize_text(self, text):
        inputs_one = self.tokenizer(text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt")
        one_ids = inputs_one.input_ids
        
        return one_ids
    
    def __len__(self):
        return self.subset_size  

    def __getitem__(self, idx):
        # batch_pairs = random.sample(self.data_pairs, self.jigsaw_size)
        
        pair_ids = random.sample(list(self.pair_indices), self.jigsaw_size)

        data_tuples = []
        for pair_id in pair_ids:
            data_tuples.append([
                        os.path.join(self.data_paths, f"{pair_id}_subject.png"),
                        os.path.join(self.data_paths, f"{pair_id}_target.png"),
                        os.path.join(self.data_paths, f"{pair_id}_prompts.txt")
                    ])

        left_images, right_images = [], []
        left_descriptions, right_descriptions = [], []
        
        for subject_image_path, target_image_path, prompts_file_path in data_tuples:
            left_image = Image.open(target_image_path).convert("RGB")  # 确保是 RGB
            right_image = Image.open(subject_image_path).convert("RGB")
            
            with open(prompts_file_path, "r") as prompts:
                lines = prompts.readlines()
                
            left_descriptions.append(lines[0].strip())
            right_descriptions.append(lines[1].strip())

            left_images.append(left_image)
            right_images.append(right_image)
        
        # **STEP 1: 先拼接 4 张 left_image（未 transform）**
        grid_size = 2  # 2x2 组合
        img_size = left_images[0].size[0] // 2  # 计算缩小后的尺寸
        resized_left_images = [img.resize((img_size, img_size), Image.BILINEAR) for img in left_images]
           
        # 创建 2x2 拼接目标图像
        target_image = Image.new("RGB", (img_size * 2, img_size * 2))
        target_image.paste(resized_left_images[0], (0, 0))
        target_image.paste(resized_left_images[1], (img_size, 0))
        target_image.paste(resized_left_images[2], (0, img_size))
        target_image.paste(resized_left_images[3], (img_size, img_size))

        if random.random() < 0.3 and self.args.with_rotation:
            target_image = random_rotation(target_image, max_angle=150)

        # **STEP 3: 统一对拼接后的 target_image 做 transform**
        target_image = self.transform(target_image)
        subject_images =[self.transform(img) for img in right_images]  # 重新 transform 右图
        subject_images = torch.stack(subject_images)
        
        # format descriptsions
        left_descripstion_combined = f"A four-panel collage image. Top left image: {left_descriptions[0]}, Top right image: {left_descriptions[1]}, bottom left image: {left_descriptions[2]}, bottom right image: {left_descriptions[3]}"
        
        one_ids = self.tokenize_text(left_descripstion_combined)
        sub_one_ids = self.tokenize_text(right_descriptions)
        sample = {
            "target_prompt": left_descripstion_combined,
            "subject_prompt": right_descriptions,
            "target_image": target_image,  # 2x2 拼接后的左图
            "subject_images": subject_images,  # 4 张右图
            "input_ids": one_ids,
            "subject_input_ids": sub_one_ids,
            "dataset_name": "subject200k_jigsaw",
            "padding_num": 0,
        }
        
        return sample



if __name__ == "__main__":
    train_transforms = transforms.Compose(
        [
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    tokenizer = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0",subfolder="tokenizer",revision=None,use_fast=False, local_files_only=True)
    tokenizer_two = AutoTokenizer.from_pretrained("stabilityai/stable-diffusion-xl-base-1.0", subfolder="tokenizer_2", revision=None, use_fast=False, local_files_only=True)
    
    dataset = Subject200k_dataset_sdxl_jigsaw_sdxl(
        data_paths="data/Subjects200K_collection3/extracted_pairs",
        transform=train_transforms,
        tokenizer=tokenizer,
        tokenizer_two=tokenizer_two
    )

    dataloader = DataLoader(dataset, jigsaw_size=1, shuffle=True)

    for i, sample in enumerate(tqdm(dataloader, desc="Processing Samples")):
        print(f"Batch {i}: Target Image Shape {sample['target_image'].shape}, Subject Images Shape {sample['subject_images'].shape}")