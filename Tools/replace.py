import os
import re

# 设置你的视频文件所在的文件夹
folder = r"D:\socilogy"

for filename in os.listdir(folder):
    # 路径拼接
    old_path = os.path.join(folder, filename)

    # 确保是文件，不是文件夹
    if os.path.isfile(old_path):
        # 匹配 "#数字"
        match = re.search(r"#(\d+)", filename)
        if match:
            number = match.group(1)  # 提取数字
            # 删除 "#" + 数字
            new_name = re.sub(r"#\d+", "", filename)
            # 新的文件名：数字-原文件名
            new_name = f"{number}-{new_name}"

            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")
