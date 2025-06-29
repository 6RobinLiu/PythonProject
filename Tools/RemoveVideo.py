import os

# 设置你的视频文件夹路径
folder_path = 'D:\NewDownLoad'  # 例如 r'D:\Videos'

# 遍历文件夹中的所有文件
all_files = os.listdir(folder_path)

# 创建一个集合，用于存放生成的文件名（去掉"【卡卡】"前缀）
generated_set = set()

for file in all_files:
    if file.startswith("【卡卡】"):
        original_name = file.replace("【卡卡】", "", 1)
        generated_set.add(original_name)

# 遍历原始文件，如果已经生成了对应的文件，则删除原视频
for file in all_files:
    if not file.startswith("【卡卡】") and file in generated_set:
        file_path = os.path.join(folder_path, file)
        try:
            os.remove(file_path)
            print(f"已删除原视频: {file}")
        except Exception as e:
            print(f"删除失败: {file}，原因: {e}")
