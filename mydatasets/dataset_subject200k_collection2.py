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
from datasets import load_dataset

class Subject200k_dataset_parquet_collection2(Dataset):
    def __init__(self, transform=None, max_len=4, tokenizer_one=None, tokenizer_two=None, args=None, subset_size=None, subset = None, collection=3):
        self.transform = transform
        self.subset_size = subset_size
        self.df = self._load_parquet_files()


    def _load_parquet_files(self):
        dataset = load_dataset('hf_cache/Yuanshi--Subjects200K')
        def filter_func(item):
            if item.get("collection") != "collection_2":
                return False
            if not item.get("quality_assessment"):
                return False
            return all(
                item["quality_assessment"].get(key, 0) >= 5
                for key in ["compositeStructure", "objectConsistency", "imageQuality"]
            )
        collection_2_valid = dataset["train"].filter(
            filter_func,
            num_proc=16,
            cache_file_name="data/collection_2_valid.arrow", # Optional
        )
        return collection_2_valid

    
    def __len__(self):
        return self.subset_size if self.subset_size else len(self.df)
        # return len(self.data_pairs) if self.subset_size is None else self.subset_size

    def __getitem__(self, idx):
        row = self.df[idx % len(self.df)]

        image = row['image']
        width, height = image.size
        left_image = image.crop((0, 0, width // 2, height))
        right_image = image.crop((width // 2, 0, width, height))

        right_description = row['description']['item']
        left_description = right_description[:-1] + " " + row['description']['description_0']
        
        left_image = self.transform(left_image)
        right_image = self.transform(right_image)

        sample = {
            "target_image": left_image,
            "subject_images": right_image.unsqueeze(0),
            "target_prompt": left_description,
            "subject_prompt": right_description,
            "dataset_name": "subject200k",
        }

        return sample

if __name__ == "__main__":

    train_transforms = transforms.Compose(
        [
            transforms.Resize((512, 512), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )

    dataset = Subject200k_dataset_parquet_collection2(transform=train_transforms)
    dataset[0]