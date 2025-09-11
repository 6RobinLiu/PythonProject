import os
import re

# 你要处理的文件夹路径
folder = r"D:\Ruby"

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    # 只处理文件，不处理子目录
    if not os.path.isfile(old_path):
        continue

    # 用正则提取末尾数字
    match = re.match(r"^(.*)\s+(\d+)$", os.path.splitext(filename)[0])
    if match:
        name_part = match.group(1).strip()   # 文件名主体
        num_part = match.group(2)            # 尾部数字
        ext = os.path.splitext(filename)[1]  # 文件扩展名

        # 生成新文件名：数字-主体 + 扩展名
        new_filename = f"{num_part}-{name_part}{ext}"
        new_path = os.path.join(folder, new_filename)

        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")
