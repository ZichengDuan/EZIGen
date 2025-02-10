from datasets import load_dataset
from datasets import concatenate_datasets, Dataset
import datasets
import os
import glob
# os.environ['HF_DATASETS_OFFLINE '] = "1"
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HOME"] = "/dev/null"  # 防止访问缓存目录

# # Load dataset
# dataset = load_dataset('Yuanshi/Subjects200K_collection3', download_mode="force_redownload")
# 获取所有本地 Parquet 文件
# parquet_files = glob.glob("/hpcfs/users/a1901664/huggingface_cache/datasets/Subjects200K_collection3/data/train-*.parquet")

# 逐个加载并合并
# datasets = [Dataset.from_parquet(f) for f in parquet_files]
# dataset = concatenate_datasets(datasets, local_files_only=True)
# dataset = load_dataset("parquet", data_files="/hpcfs/users/a1901664/huggingface_cache/datasets/Subjects200K_collection3/data/train-00004-of-00012.parquet")
import pandas as pd
import glob

# 获取所有 parquet 文件
parquet_files = glob.glob("/hpcfs/users/a1901664/huggingface_cache/datasets/Subjects200K_collection3/data/train-*.parquet")

# 读取所有文件并合并
df = pd.concat([pd.read_parquet(f) for f in parquet_files], ignore_index=True)

print(df.head())
print()
# dataset_folder = "/hpcfs/users/a1901664/huggingface_cache/datasets/Yuanshi___subjects200_k/default/0.0.0/622a7cf65a222fcb"
# arrow_files = os.listdir(dataset_folder)

# dataset = concatenate_datasets([Dataset.from_file(os.path.join(dataset_folder, arrow_file)) for arrow_file in arrow_files if ".arrow" in arrow_file])

# def filter_func(item):
#     if item.get("collection") != "collection_3":
#         return False
#     if not item.get("quality_assessment"):
#         return False
#     return all(
#         item["quality_assessment"].get(key, 0) >= 5 for key in ["objectConsistency"]
#     )


# for i in range(len(dataset)):
#     if dataset[-(i+1)].get("collection") is not None and "3" in dataset[-(i+1)].get("collection"):
#         print(dataset[-(i+1)].get("collection"))


# collection_3_valid = dataset.filter(
#     filter_func,
#     num_proc=16,
#     # cache_file_name="./cache/dataset/collection_3_valid.arrow", # Optional
# )



# collection_3_valid
# breakpoint()

