import os
import re

# 设置你的文件夹路径
folder = r"C:\VideoDownLoad3"

for filename in os.listdir(folder):
    old_path = os.path.join(folder, filename)

    if os.path.isfile(old_path) and filename.lower().endswith(".mp4"):
        # 1. 去掉 "： Crash Course Entrepreneurship" 及后面部分
        base_name = re.sub(r"： Crash Course Entrepreneurship.*?(#\d+)", r"\1", filename)

        # 2. 提取 "#数字"
        match = re.search(r"#(\d+)", base_name)
        if match:
            number = match.group(1)
            # 去掉 "#数字"
            new_name = re.sub(r"#\d+", "", base_name).strip()
            # 拼接新文件名
            new_name = f"{number}-{new_name}"

            # 确保扩展名正确
            if not new_name.lower().endswith(".mp4"):
                new_name += ".mp4"

            new_path = os.path.join(folder, new_name)
            os.rename(old_path, new_path)
            print(f"重命名: {filename} -> {new_name}")
