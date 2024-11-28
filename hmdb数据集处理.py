import os
import shutil
'''
    将现有数据集转换为新的数据集格式,
    现有数据集格式:
    hmdb51/
        class1/
            video1.avi
            video2.avi
        class2/
            video1.avi
            video2.avi
        ...
'''

# 设置现有数据集的根目录和目标目录
source_root = '/mnt/SSD8T/home/huangwei/projects/FROSTER/data/hmdb51-origin'
target_root = '/mnt/SSD8T/home/huangwei/projects/FROSTER/data/hmdb51'

# 确保目标目录存在
os.makedirs(target_root, exist_ok=True)

# 遍历每个类别文件夹
for category in os.listdir(source_root):
    category_path = os.path.join(source_root, category)
    
    # 确保是目录
    if os.path.isdir(category_path):
        # 遍历类别文件夹中的每个文件
        for video_file in os.listdir(category_path):
            source_file = os.path.join(category_path, video_file)
            target_file = os.path.join(target_root, video_file)
            
            # 复制文件到目标目录
            shutil.copy2(source_file, target_file)
            print(f"Copied {source_file} to {target_file}")

print("All files have been copied.")