import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import io
import os
from PIL import Image
import torchvision.transforms as transforms
from tqdm import tqdm
import json
from datasets import load_dataset
import argparse


class Subject200k_dataset_parquet_collection3(Dataset):
    def __init__(self, data_paths=None, parquet_paths=None, transform=None, max_len=4, tokenizer_one=None, tokenizer_two=None, args=None, subset_size=None, subset = None, collection=3):
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

    def _load_parquet_files(self, subset):
        files = [os.path.join(self.parquet_paths, f) for f in os.listdir(self.parquet_paths) if f.endswith('.parquet')]
        df_list = []
        for i, f in enumerate(tqdm(files, desc="Loading Parquet Files")):
            if not (i >= subset[0] and i < subset[1]):
                continue
            df_list.append(pd.read_parquet(f))
        
        return pd.concat(df_list, ignore_index=True)

    
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
        breakpoint()
        left_desciption = raw_json['image_descriptions']['descriptive_style']['left_image']
        right_desciption = raw_json['image_descriptions']['pronoun_style']['right_image']
        
        sample = {
          "left_img": left_image,
          "right_img": right_image,
          "left_desciption": left_desciption,
          "right_desciption": right_desciption,
        }

        return sample


class Subject200k_dataset_parquet_collection2(Dataset):
    def __init__(self, data_paths=None, parquet_paths=None, transform=None, max_len=4, tokenizer_one=None, tokenizer_two=None, args=None, subset_size=None, subset = None, collection=3):
        self.data_paths = data_paths
        self.parquet_paths = parquet_paths
        self.transform = transform
        self.df = self._load_parquet_files()
        self.args = args
        self.max_len = max_len
        self.tokenizer_one = tokenizer_one
        self.tokenizer_two = tokenizer_two


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
        return len(self.df)
        # return len(self.data_pairs) if self.subset_size is None else self.subset_size

    def __getitem__(self, idx):
        row = self.df[idx]
        image = row['image']
        width, height = image.size
        left_image = image.crop((0, 0, width // 2, height))
        right_image = image.crop((width // 2, 0, width, height))

        # # 查找白色分界线
        # dividing_line = None
        # for x in range(width):
        #     if abs(x - width // 2) > width // 6:  # 确保分界线在图像中线左右1/6的位置
        #         continue
            
        #     white_count = 0
        #     for y in range(height):
        #         r, g, b = image.getpixel((x, y))
        #         if r > 190 and g > 190 and b > 190:  # 假设白色背景是接近(255,255,255)的
        #             white_count += 1
                    
        #     if white_count / height >= 0.95:
        #         dividing_line = x
        #         break
                
        # if dividing_line is not None:
        #     left_image = image.crop((0, 0, dividing_line, height)).resize((1024, 1024))
        #     right_image = image.crop((dividing_line, 0, width, height)).resize((1024, 1024))
        # else:
        #   left_image = None
        #   right_image = None

        right_desciption = row['description']['item']
        left_desciption = right_desciption[:-1] + " " + row['description']['description_0']
        sample = {
          "left_img": left_image,
          "right_img": right_image,
          "left_desciption": left_desciption,
          "right_desciption": right_desciption,
        }

        return sample

import os
import shutil
from glob import glob

def consolidate_pairs(src_root: str, dst_root: str):
    """
    Consolidate paired files from multiple 'pairs[1-6]' folders into a single target folder,
    renaming them with continuous indices.

    Args:
        src_root (str): Path to the root directory containing 'pairs1' to 'pairs6' subfolders.
        dst_root (str): Path to the output directory where consolidated files will be saved.
    """
    os.makedirs(dst_root, exist_ok=True)
    index = 0

    for i in range(1, 7):
        folder = os.path.join(src_root, f'pairs{i}')
        target_files = sorted(glob(os.path.join(folder, '*_target.png')))

        for target_path in target_files:
            print(index)
            prefix = os.path.basename(target_path).split('_')[0]

            subject_path = os.path.join(folder, f"{prefix}_subject.png")
            prompt_path = os.path.join(folder, f"{prefix}_prompts.txt")

            if not (os.path.exists(subject_path) and os.path.exists(prompt_path)):
                print(f"[Warning] Skipped incomplete pair with prefix '{prefix}' in {folder}")
                continue

            shutil.copy2(target_path, os.path.join(dst_root, f"{index}_target.png"))
            shutil.copy2(subject_path, os.path.join(dst_root, f"{index}_subject.png"))
            shutil.copy2(prompt_path, os.path.join(dst_root, f"{index}_prompts.txt"))
            index += 1

    print(f"✅ Done. Total {index} complete pairs copied to '{dst_root}'.")

# 示例调用
# consolidate_pairs("data/Subjects200K_collection3", "data/Subjects200K_collection3/extracted_pairs")


def extract_subject200k_subset_collection3(parquet_root: str, extraction_base_path: str, subset_idx: int):
    """
    Extracts a subset of the Subject200K dataset and saves paired images + prompt text to disk.

    Args:
        parquet_root (str): Path to the root directory containing parquet files.
        extraction_base_path (str): Directory under which pairsX folders are located.
        subset_idx (int): Index for subset range from predefined subsets list (0-5).
    """
    subsets = [[0, 2], [2, 4], [4, 6], [6, 8], [8, 10], [10, 12]]

    train_transforms = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = Subject200k_dataset_sdxl_parquet(
        parquet_paths=parquet_root,
        transform=train_transforms,
        tokenizer_one=None,
        tokenizer_two=None,
        subset=subsets[subset_idx]
    )

    extraction_path = os.path.join(extraction_base_path, f"pairs{subset_idx + 1}")
    os.makedirs(extraction_path, exist_ok=True)

    # Determine current max prefix
    existing_files = os.listdir(extraction_path)
    if existing_files:
        max_prefix = int(sorted(existing_files, key=lambda x: int(x.split("_")[0]))[::-1][0].split("_")[0])
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
        breakpoint()
        file_prefix = max_prefix + i - problematic
        left_image.save(os.path.join(extraction_path, f"{file_prefix}_target.png"))
        right_image.save(os.path.join(extraction_path, f"{file_prefix}_subject.png"))

        with open(os.path.join(extraction_path, f"{file_prefix}_prompts.txt"), "w") as f:
            f.write(left_desciption + "\n")
            f.write(right_desciption)

    print(f"✅ Subset {subset_idx} extraction complete. Problematic samples skipped: {problematic}")


def extract_subject200k_subset_collection2(parquet_root: str, extraction_base_path: str, subset_idx: int):
    """
    Extracts a subset of the Subject200K dataset and saves paired images + prompt text to disk.

    Args:
        parquet_root (str): Path to the root directory containing parquet files.
        extraction_base_path (str): Directory under which pairsX folders are located.
        subset_idx (int): Index for subset range from predefined subsets list (0-5).
    """

    train_transforms = transforms.Compose([
        transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
        transforms.Normalize([0.5], [0.5]),
    ])

    dataset = Subject200k_dataset_parquet_collection2(
        parquet_paths=parquet_root,
        transform=train_transforms,
        tokenizer_one=None,
        tokenizer_two=None,
    )

    extraction_path = os.path.join(extraction_base_path, f"extracted_pairs")
    os.makedirs(extraction_path, exist_ok=True)

    # Determine current max prefix
    existing_files = os.listdir(extraction_path)
    if existing_files:
        max_prefix = int(sorted(existing_files, key=lambda x: int(x.split("_")[0]))[::-1][0].split("_")[0])
    else:
        max_prefix = 0

    problematic = 0
    for i in tqdm(range(len(dataset)), desc="Processing Samples"):
        sample = dataset[i]
        left_image = sample['left_img']
        right_image = sample['right_img']
        left_desciption = sample['left_desciption']
        right_desciption = sample['right_desciption']

        if left_image is None:
            problematic += 1
            continue
        file_prefix = max_prefix + i - problematic
        left_image.save(os.path.join(extraction_path, f"{file_prefix}_target.png"))
        right_image.save(os.path.join(extraction_path, f"{file_prefix}_subject.png"))

        with open(os.path.join(extraction_path, f"{file_prefix}_prompts.txt"), "w") as f:
            f.write(left_desciption + "\n")
            f.write(right_desciption)

    print(f"✅ Subset {subset_idx} extraction complete. Problematic samples skipped: {problematic}")


# 可选 CLI 封装
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_idx', type=int, default=0, help='Subset index (0-5)')
    parser.add_argument('--parquet_root', type=str, default='hf_cache/Yuanshi--Subjects200K/data')
    parser.add_argument('--extraction_base', type=str, default='data/Subjects200K_collection2')
    parser.add_argument('--extraction_dst', type=str, default='data/Subjects200K_collection3/extracted_pairs')
    args = parser.parse_args()

    extract_subject200k_subset_collection2(args.parquet_root, args.extraction_base, args.subset_idx)
    # consolidate_pairs(args.extraction_base, args.extraction_dst)