import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import io
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json


class Subject200k_dataset(Dataset):
    def __init__(self, data_paths, transform=None, max_len=4, tokenizer_one=None, tokenizer_two=None, args=None, subset_size=None):
        self.data_paths = data_paths
        self.transform = transform
        # self.df = self._load_parquet_files()
        self.data_pairs = self._load_image_pairs()
        self.args = args
        self.max_len = max_len
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two
        self.subset_size = subset_size

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
    
    def _load_parquet_files(self):
        files = [os.path.join(self.data_paths, f) for f in os.listdir(self.data_paths) if f.endswith('.parquet')]
        df_list = []
        for i, f in enumerate(tqdm(files, desc="Loading Parquet Files")):
            if i < 8:
                continue
            df_list.append(pd.read_parquet(f))
        
        return pd.concat(df_list, ignore_index=True)

    def _load_image_pairs(self):
        data_tuples = []
        idx = 0
        while True:
            if os.path.exists(os.path.join(self.data_paths, f"{idx}_subject.png")):
                data_tuples.append([os.path.join(self.data_paths, f"{idx}_subject.png"), os.path.join(self.data_paths, f"{idx}_target.png"), os.path.join(self.data_paths, f"{idx}_prompts.txt")])
                idx += 1
            else:
                break
        return data_tuples
    
    def __len__(self):
        # return len(self.df) if self.subset_size is None else self.subset_size
        return len(self.data_pairs) if self.subset_size is None else self.subset_size

    def __getitem__(self, idx):
        try:
            subject_image_path, target_image_path, prompts_file_path = self.data_pairs[idx % len(self.data_pairs)]
        except:
            print(idx % len(self.data_pairs), len(self.data_pairs))
            
        left_image = Image.open(target_image_path)
        right_image = Image.open(subject_image_path)
        
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
    
    dataset = Subject200k_dataset(
        data_paths="data/Subjects200K_collection3/extracted_pairs",
        transform=train_transforms,
        tokenizer_one=None,
        tokenizer_two=None
    )

    for i in tqdm(range(len(dataset.data_pairs)), desc="Processing Samples"):
        sample = dataset[i]
        # print(sample["item"], sample["target_prompt"])