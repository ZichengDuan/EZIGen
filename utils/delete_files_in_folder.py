import os
from pathlib import Path
from tqdm import tqdm

def delete_files_in_folder(folder_path):
    # 将传入的路径转换为Path对象
    path = Path(folder_path)

    # 检查给定路径是否是一个存在的目录
    if not path.exists() or not path.is_dir():
        print("提供的路径不是一个有效的文件夹")
        return
    
    # 获取文件夹中所有的文件和子文件夹列表
    items = list(path.iterdir())

    # 如果文件夹为空，直接返回
    if not items:
        print("文件夹为空，无需删除任何文件")
        return

    # 使用tqdm创建进度条
    for item in tqdm(items, desc='删除进度', unit='项'):
        if item.is_file():
            try:
                item.unlink()  # 删除文件
            except Exception as e:
                print(f"无法删除 {item}: {e}")
        elif item.is_dir():
            try:
                # 对于子文件夹，递归调用本函数
                delete_files_in_folder(item)
                # 只清空子文件夹的内容，不删除子文件夹本身
            except Exception as e:
                print(f"处理子文件夹 {item} 时出错: {e}")

# 示例用法：
delete_files_in_folder('/mnt/sh_nas/duanzicheng.dzc/Data/VITON-HD/test')
