import os

# 替换为你的文件夹路径
folder_path = r"D:\Machine Learning Hardware and Systems"

for filename in os.listdir(folder_path):
    old_path = os.path.join(folder_path, filename)

    if os.path.isfile(old_path):
        name, ext = os.path.splitext(filename)

        # 删除所有的【卡卡】
        new_name = name.replace("【卡卡】", "").strip()

        new_filename = new_name + ext
        new_path = os.path.join(folder_path, new_filename)

        os.rename(old_path, new_path)
        print(f"重命名: {filename} -> {new_filename}")
