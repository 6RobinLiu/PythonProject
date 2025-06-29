import json

# 打开 JSON 文件
with open("playlist.json", "r", encoding="utf-8") as f:
    data = json.load(f)

entries = data["entries"]

# 提取第121到140个（注意：Python下标从0开始）
selected_entries = entries[101:120]

with open("video_urls.txt", "w", encoding="utf-8") as f:
    for entry in selected_entries:
        url = entry["url"]
        f.write(url + "\n")

print("已提取链接并保存至 video_urls.txt")
