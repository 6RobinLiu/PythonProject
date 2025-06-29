input_file = "video_urls.txt"       # 你已经有的包含 300 行的链接文件
output_file = "todown.txt"   # 输出目标文件

with open(input_file, "r", encoding="utf-8") as f:
    lines = f.readlines()

# 行号从 0 开始，所以 101~120 是 lines[100:120]
subset = lines[100:120]

with open(output_file, "w", encoding="utf-8") as f:
    f.writelines(subset)

print(f"提取完成，共 {len(subset)} 个链接已保存到 {output_file}")
