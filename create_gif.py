# create_gif.py

import os
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from PIL import Image

def create_gif(start_step, end_step, output_gif_path, image_folder='output_images'):
    """
    从图片文件夹中读取指定步骤的图片并生成GIF文件
    :param start_step: 起始步骤
    :param end_step: 结束步骤
    :param output_gif_path: 输出GIF文件路径
    :param image_folder: 图片文件夹路径，默认是'output_images'
    """
    images = []
    
    # 遍历指定范围的step值
    for step in range(start_step, end_step + 1):
        # 构建图片文件名
        image_filename = os.path.join(image_folder, f"highway_step_{step}.png")
        
        # 检查文件是否存在
        if os.path.exists(image_filename):
            # 读取并打开图片
            img = Image.open(image_filename)
            images.append(img)
        else:
            print(f"Warning: Image for step {step} not found.")
    
    # 如果没有有效的图片，则退出
    if not images:
        print("No valid images found to create GIF.")
        return
    
    # 保存为GIF文件
    images[0].save(output_gif_path, save_all=True, append_images=images[1:], duration=500, loop=0)
    print(f"GIF saved to {output_gif_path}")

# 示例调用：生成从step 0到step 200的GIF
create_gif(start_step=1, end_step=200, output_gif_path="highway_simulation.gif")
