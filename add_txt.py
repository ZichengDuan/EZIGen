import os

folder_path = "data/Subjects200K_collection3/extracted_pairs"

# 检查路径是否存在
if not os.path.exists(folder_path):
    print(f"❌ 路径不存在: {folder_path}")
else:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # 仅处理没有后缀的文件
        if os.path.isfile(file_path) and '.' not in filename:
            new_file_path = file_path + ".txt"
            os.rename(file_path, new_file_path)
            print(f"Renamed: {filename} -> {filename}.txt")

    print("✅ 所有没有后缀的文件已补充 .txt 格式")