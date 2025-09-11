import os
import re

# 指定文件夹路径
folder_path = r"C:\VideoDownload"  # 改成你的视频文件夹路径，比如 r"D:\Videos"

# 正则模式：删除前缀和后缀的 [xxx]
prefix = r"^Stanford CS25： V5 I\s*"    # 匹配开头
suffix = r"\s*\[.*?\]$"                 # 匹配末尾的 [xxx]

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if os.path.isfile(old_path):
        name, ext = os.path.splitext(filename)

        # 删除前缀和后缀
        new_name = re.sub(prefix, '', name)
        new_name = re.sub(suffix, '', new_name)

        new_filename = new_name.strip() + ext
        new_path = os.path.join(folder_path, new_filename)

        # 重命名文件
        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")
