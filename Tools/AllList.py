'''
对于youtube的大播放列表   会有100以后的相关信息  无法通过yt-dlp获取
所以就用访问自己刷新得到视频信息   通过前端的element获取
然后处理复制得到的对应元素的txt
'''

from bs4 import BeautifulSoup

input_file = "yt.txt"
output_file = "video_urls.txt"

with open(input_file, "r", encoding="utf-8") as f:
    html = f.read()

soup = BeautifulSoup(html, "html.parser")
video_links = set()

for a in soup.find_all("a"):
    href = a.get("href")
    if href and "/watch?v=" in href and "list=" in href:
        # 清洗链接，去掉 index、pp 等多余参数
        full_url = "https://www.youtube.com" + href.split("&")[0]
        video_links.add(full_url)

# 写入输出文件
with open(output_file, "w", encoding="utf-8") as f:
    for link in sorted(video_links):
        f.write(link + "\n")

print(f"提取完成，共 {len(video_links)} 个视频链接，已保存到 {output_file}")
