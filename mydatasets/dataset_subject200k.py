import sys
sys.path.append("..")
sys.path.append(".")
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import io
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import json
from utils.image_processors import random_horizontal_flip, random_rotation, random_scaling
import numpy as np

class Subject200k_dataset_sdxl(Dataset):
    def __init__(self, data_paths=None, parquet_paths=None, transform=None, max_len=4, tokenizer_one=None, tokenizer_two=None, args=None, subset_size=None):
        self.data_paths = data_paths
        self.parquet_paths = parquet_paths
        self.transform = transform
        self.transform = transform
        self.pair_indices = np.load("data/Subjects200K_collection3/unique_idxs.npy")
        self.args = args
        self.max_len = max_len
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.subset_size = subset_size if subset_size is not None else 50000

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
    
    # def _load_parquet_files(self):
    #     files = [os.path.join(self.parquet_paths, f) for f in os.listdir(self.parquet_paths) if f.endswith('.parquet')]
    #     df_list = []
    #     for i, f in enumerate(tqdm(files, desc="Loading Parquet Files")):
    #         if i > 1:
    #             continue
    #         df_list.append(pd.read_parquet(f))
        
    #     return pd.concat(df_list, ignore_index=True)

    # def _load_image_pairs(self):
    #     data_tuples = []
    #     idx = 0
    #     while True:
    #         if os.path.exists(os.path.join(self.data_paths, f"{idx}_subject.png")):
    #             data_tuples.append([os.path.join(self.data_paths, f"{idx}_subject.png"), os.path.join(self.data_paths, f"{idx}_target.png"), os.path.join(self.data_paths, f"{idx}_prompts.txt")])
    #             idx += 1
    #         else:
    #             break
    #     return data_tuples
    
    def __len__(self):
        # return len(self.df) if self.subset_size is None else self.subset_size
        return self.subset_size if self.subset_size else len(self.pair_indices)


    def __getitem__(self, idx):
        idx = self.pair_indices[idx % len(self.pair_indices)]

        subject_image_path, target_image_path, prompts_file_path = os.path.join(self.data_paths, f"{idx}_subject.png"), os.path.join(self.data_paths, f"{idx}_target.png"), os.path.join(self.data_paths, f"{idx}_prompts.txt")
            
        left_image = Image.open(target_image_path)
        right_image = Image.open(subject_image_path)

        if random.random() < 0.3:
            left_image, right_image = random_horizontal_flip(left_image, right_image)

        # augment right subject image
        if self.args.with_rotation:
            right_image = random_rotation(right_image, max_angle=150)

        if random.random() < 0.3 and self.args.with_scale:
            right_image = random_scaling(right_image)

        with open(prompts_file_path, "r") as prompts:
            lines = prompts.readlines()
            
        left_description = lines[0].strip()
        right_description = lines[1].strip()
        
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        else:
            left_image, right_image = None, None
        
        if random.randint(1,100) / 100 < self.args.drop_text_ratio:
            left_description = ""

        # tokenize
        if self.tokenizer_one:
            one_ids, two_ids = self.tokenize_text(left_description)
            sub_one_ids, sub_two_ids = self.tokenize_text(right_description)
        else:
            one_ids, two_ids, sub_one_ids, sub_two_ids = None, None, None, None
        
        sample = {
            "target_prompt": left_description,
            "subject_prompt": right_description,
            # dzc
            "target_image": left_image,
            "subject_images": right_image.unsqueeze(0),
            "input_ids": one_ids,
            "input_ids_two": two_ids,
            "subject_input_ids": sub_one_ids,
            "subject_input_ids_two": sub_two_ids,
            "dataset_name": "subject200k",
            "padding_num": self.args.num_sub_img - 1,
        }
        
        return sample



class Subject200k_dataset_sd21(Dataset):
    def __init__(self, data_paths=None, parquet_paths=None, transform=None, max_len=4, tokenizer=None, args=None, subset_size=None):
        self.data_paths = data_paths
        self.parquet_paths = parquet_paths
        self.transform = transform
        self.transform = transform
        self.pair_indices = np.load("data/Subjects200K_collection3/unique_idxs.npy")
        self.args = args
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.subset_size = subset_size if subset_size is not None else 50000

    def tokenize_text(self, text):
        inputs_one = self.tokenizer(
            text, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )
        one_ids = inputs_one.input_ids
        
        return one_ids
    
    def __len__(self):
        # return len(self.df) if self.subset_size is None else self.subset_size
        return self.subset_size if self.subset_size else len(self.pair_indices)


    def __getitem__(self, idx):
        idx = self.pair_indices[idx % len(self.pair_indices)]

        subject_image_path, target_image_path, prompts_file_path = os.path.join(self.data_paths, f"{idx}_subject.png"), os.path.join(self.data_paths, f"{idx}_target.png"), os.path.join(self.data_paths, f"{idx}_prompts.txt")
            
        left_image = Image.open(target_image_path)
        right_image = Image.open(subject_image_path)

        if random.random() < 0.3:
            left_image, right_image = random_horizontal_flip(left_image, right_image)

        # augment right subject image
        if self.args.with_rotation:
            right_image = random_rotation(right_image, max_angle=150)

        if random.random() < 0.3 and self.args.with_scale:
            right_image = random_scaling(right_image)

        with open(prompts_file_path, "r") as prompts:
            lines = prompts.readlines()
            
        left_description = lines[0].strip()
        right_description = lines[1].strip()
        
        if self.transform:
            left_image = self.transform(left_image)
            right_image = self.transform(right_image)
        else:
            left_image, right_image = None, None
        
        
        # tokenize
        if self.tokenizer:
            one_ids = self.tokenize_text(left_description)
            sub_one_ids = self.tokenize_text(right_description)
        else:
            one_ids, sub_one_ids= None, None
        
        sample = {
            "target_prompt": left_description,
            "subject_prompt": right_description,
            # dzc
            "target_image": left_image,
            "subject_images": right_image.unsqueeze(0),
            "input_ids": one_ids,
            "subject_input_ids": sub_one_ids,
            "dataset_name": "subject200k",
            "padding_num": self.args.num_sub_img - 1,
        }
        
        return sample




    # def __getitem__(self, idx):
    #     try:
    #         subject_image_path, target_image_path, prompts_file_path = self.data_pairs[idx % len(self.data_pairs)]
    #     except:
    #         print(idx % len(self.data_pairs), len(self.data_pairs))
            
    #     left_image = Image.open(target_image_path)
    #     right_image = Image.open(subject_image_path)

    #     left_image, right_image = random_horizontal_flip(left_image, right_image)

    #     # augment right subject image
    #     right_image = random_rotation(right_image, max_angle=150)
    #     right_image = random_scaling(right_image)

    #     with open(prompts_file_path, "r") as prompts:
    #         lines = prompts.readlines()
            
    #     left_description = lines[0].strip()
    #     right_description = lines[1].strip()
        
    #     if self.transform:
    #         left_image = self.transform(left_image)
    #         right_image = self.transform(right_image)
    #     else:
    #         left_image, right_image = None, None
        
        
    #     # tokenize
    #     if self.tokenizer_one:
    #         one_ids, two_ids = self.tokenize_text(left_description)
    #         sub_one_ids, sub_two_ids = self.tokenize_text(right_description)
    #     else:
    #         one_ids, two_ids, sub_one_ids, sub_two_ids = None, None, None, None
        
    #     sample = {
    #         "target_prompt": left_description,
    #         "subject_prompt": right_description,
    #         # dzc
    #         "target_image": left_image,
    #         "subject_images": right_image.unsqueeze(0),
    #         "input_ids": one_ids,
    #         "input_ids_two": two_ids,
    #         "subject_input_ids": sub_one_ids,
    #         "subject_input_ids_two": sub_two_ids,
    #         "dataset_name": "subject200k",
    #     }
        
    #     return sample


if __name__ == "__main__":
    train_transforms = transforms.Compose(
        [
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    dataset = Subject200k_dataset_sdxl(
        parquet_paths="/mnt/sh_nas/duanzicheng.dzc/Data/Subjects200K_collection3/data",
        transform=train_transforms,
        tokenizer_one=None,
        tokenizer_two=None
    )

    for i in tqdm(range(len(dataset.data_pairs)), desc="Processing Samples"):
        sample = dataset[i]
        # print(sample["item"], sample["target_prompt"])