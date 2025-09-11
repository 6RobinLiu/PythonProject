import os
from PIL import Image

# === 参数 ===
input_image_path = "car.png"  # 输入图片路径
output_dir = "output_parts"     # 输出文件夹

# === 创建输出文件夹 ===
os.makedirs(output_dir, exist_ok=True)

# === 打开图像 ===
img = Image.open(input_image_path)
width, height = img.size

# === 分割参数 ===
rows = 2  # 高度方向分成 2 份（短边）
cols = 3  # 宽度方向分成 3 份（长边）
tile_width = width // cols
tile_height = height // rows

# === 分割并保存 ===
count = 0
for row in range(rows):
    for col in range(cols):
        left = col * tile_width
        upper = row * tile_height
        right = (col + 1) * tile_width
        lower = (row + 1) * tile_height
        tile = img.crop((left, upper, right, lower))
        
        output_path = os.path.join(output_dir, f"part_{count}.jpg")
        tile.save(output_path)
        print(f"Saved {output_path}")
        count += 1
