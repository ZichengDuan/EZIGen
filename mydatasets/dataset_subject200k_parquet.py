import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import io
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json


class Subject200k_dataset_sdxl_parquet(Dataset):
    def __init__(self, data_paths=None, parquet_paths=None, transform=None, max_len=4, tokenizer_one=None, tokenizer_two=None, args=None, subset_size=None, subset = None):
        self.data_paths = data_paths
        self.parquet_paths = parquet_paths
        self.transform = transform
        self.df = self._load_parquet_files(subset)
        # self.data_pairs = self._load_image_pairs()
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
    
    def _load_parquet_files(self, subset):
        files = [os.path.join(self.parquet_paths, f) for f in os.listdir(self.parquet_paths) if f.endswith('.parquet')]
        df_list = []
        for i, f in enumerate(tqdm(files, desc="Loading Parquet Files")):
            if not (i >= subset[0] and i < subset[1]):
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
        return len(self.df) if self.subset_size is None else self.subset_size
        # return len(self.data_pairs) if self.subset_size is None else self.subset_size

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(io.BytesIO(row['image']['bytes']))
        width, height = image.size
        # left_image = image.crop((0, 0, width // 2, height))
        # right_image = image.crop((width // 2, 0, width, height))

        # 查找白色分界线
        dividing_line = None
        for x in range(width):
            if abs(x - width // 2) > width // 6:  # 确保分界线在图像中线左右1/6的位置
                continue
            
            white_count = 0
            for y in range(height):
                r, g, b = image.getpixel((x, y))
                if r > 190 and g > 190 and b > 190:  # 假设白色背景是接近(255,255,255)的
                    white_count += 1
                    
            if white_count / height >= 0.95:
                dividing_line = x
                break
                
        if dividing_line is not None:
            left_image = image.crop((0, 0, dividing_line, height)).resize((1024, 1024))
            right_image = image.crop((dividing_line, 0, width, height)).resize((1024, 1024))
        else:
          left_image = None
          right_image = None

        raw_json = json.loads(row['raw_json'])
        left_desciption = raw_json['image_descriptions']['descriptive_style']['left_image']
        right_desciption = raw_json['image_descriptions']['descriptive_style']['right_image']

        sample = {
          "left_img": left_image,
          "right_img": right_image,
          "left_desciption": left_desciption,
          "right_desciption": right_desciption,
        }

        return sample


if __name__ == "__main__":
    subsets = [[0, 2], [2, 4], [4, 6], [6,8], [8, 10], [10, 12]]

    train_transforms = transforms.Compose(
        [
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize([0.5], [0.5]),
        ]
    )
    
    subset_idx = 3 # [0, 1, 2, 3, 4 ,5]

    dataset = Subject200k_dataset_sdxl_parquet(
        parquet_paths="data/Subjects200K_collection3/data",
        transform=train_transforms,
        tokenizer_one=None,
        tokenizer_two=None,
        subset=subsets[subset_idx]
    )

    # calclulate the max of the file prefix
    # desired saving path
    extraction_path = f"data/Subjects200K_collection3/pairs{str(subset_idx + 1)}"
    if len(os.listdir(extraction_path)) > 0:
        max_prefix = int(sorted(os.listdir(extraction_path), key=lambda x: int(x.split("_")[0]))[::-1][0].split("_")[0])
    else:
        max_prefix = 0

    problematic = 0
    for i in tqdm(range(len(dataset.df)), desc="Processing Samples"):
        sample = dataset[i]
        left_image = sample['left_img']
        right_image = sample['right_img']
        left_desciption = sample['left_desciption']
        right_desciption = sample['right_desciption']

        if left_image is None:
            problematic += 1
            continue

        # save to local folder
        left_image.save(os.path.join(extraction_path, f"{max_prefix + i - problematic}_target.png"))
        right_image.save(os.path.join(extraction_path, f"{max_prefix + i - problematic}_subject.png"))

        with open(os.path.join(extraction_path, f"{max_prefix + i - problematic}_prompts.txt"), "w") as prompt_file:
            prompt_file.write(left_desciption + '\n')
            prompt_file.write(right_desciption)

    print(problematic)