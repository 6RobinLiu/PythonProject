import os
import re

# 替换为你的文件夹路径
folder_path = r"D:\Machine Learning Hardware and Systems"
# 匹配文件名末尾的 [任意内容]
pattern = r'\s*\[.*?\]$'

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if os.path.isfile(old_path):
        name, ext = os.path.splitext(filename)

        # 删除末尾 [xxxx]
        new_name = re.sub(pattern, '', name).strip()

        new_filename = new_name + ext
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")
