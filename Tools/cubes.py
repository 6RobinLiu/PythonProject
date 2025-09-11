import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np

# 宽、高、深
width = 16   # x-axis (横向)
height = 9   # y-axis (纵向)
depth = 7    # z-axis (向后)

cube_size = 1.0
gap = 0.1  # 控制间距，越小越紧凑

def draw_cube(ax, origin, size):
    x, y, z = origin
    v = np.array([
        [x, y, z],
        [x + size, y, z],
        [x + size, y + size, z],
        [x, y + size, z],
        [x, y, z + size],
        [x + size, y, z + size],
        [x + size, y + size, z + size],
        [x, y + size, z + size]
    ])
    faces = [[0,1,2,3], [4,5,6,7], [0,1,5,4],
             [2,3,7,6], [1,2,6,5], [0,3,7,4]]
    ax.add_collection3d(
        Poly3DCollection([v[face] for face in faces],
                         facecolors='lightgrey',
                         edgecolors='black',
                         linewidths=0.1)
    )

# 创建图像
fig = plt.figure(figsize=(14, 10))
ax = fig.add_subplot(111, projection='3d')

# 构造 16×9×7 的体素阵列
for z in range(depth):     # 向后堆叠
    for y in range(height): # 上下
        for x in range(width): # 左右
            dx = x * (cube_size + gap)
            dy = y * (cube_size + gap)
            dz = z * (cube_size + gap)
            draw_cube(ax, (dx, dy, dz), cube_size)

# 设置立体视角
ax.set_box_aspect([width, height, depth])
ax.view_init(elev=30, azim=-60)  # elev 抬头角度，azim 横向旋转

ax.set_axis_off()
plt.tight_layout()
plt.savefig("perspective_cube_wall.png", dpi=300)
plt.show()
