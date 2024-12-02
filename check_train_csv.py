import os
from pathlib import Path
import csv
import argparse

'''
check_train_csv.py

该脚本用于检查 train_1.csv 文件中的视频文件是否存在于数据集中。
如果发现缺失的文件，将会输出到控制台，并根据命令行参数选择是否将这些文件从 CSV 中删除。

使用方法：
    python check_train_csv.py [--remove]

参数：
    --remove    如果指定，该脚本将删除 CSV 文件中不存在的视频条目。
'''

def parse_train_csv(csv_path):
    """
    解析 @test.csv 文件，提取视频文件名和标签。
    """
    video_files = []
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            for row in reader:
                if not row:
                    continue
                if len(row) < 2:
                    continue
                filename = row[0].strip()
                label = row[1].strip()
                video_files.append((filename, label))
    except Exception as e:
        print(f"解析 CSV 文件时出错: {e}")
    return video_files

def get_all_video_files(data_root):
    """
    递归获取数据集目录下的所有视频文件。
    """
    video_files = []
    for root, _, files in os.walk(data_root):
        for file in files:
            if file.endswith('.avi'):  # 假设视频文件是 .avi 格式
                video_files.append(os.path.relpath(os.path.join(root, file), data_root))
    return video_files

def check_videos_exist(data_root, video_files):
    """
    检查视频文件是否存在于数据集中。
    """
    all_video_files = get_all_video_files(data_root)
    missing_files = []
    for filename, label in video_files:
        if filename not in all_video_files:
            missing_files.append((filename, label))
    return missing_files

def remove_missing_from_csv(csv_path, video_files, missing_files):
    """
    从 CSV 文件中移除缺失的视频文件条目。

    参数：
        csv_path (Path): train_1.csv 文件的路径
        video_files (list of tuples): 原始的视频文件列表
        missing_files (list of tuples): 缺失的视频文件列表
    """
    # 创建缺失文件的集合以加快查找速度
    missing_set = set(filename for filename, _ in missing_files)
    
    # 过滤掉缺失的文件
    updated_video_files = [ (filename, label) for filename, label in video_files if filename not in missing_set ]
    
    try:
        # 备份原始 CSV 文件
        backup_path = csv_path.with_suffix('.csv.bak')
        os.rename(csv_path, backup_path)
        print(f"已备份原始 CSV 文件到 {backup_path}")
        
        # 写入更新后的 CSV 文件
        with open(csv_path, 'w', encoding='utf-8', newline='') as f:
            writer = csv.writer(f, delimiter=',')
            for filename, label in updated_video_files:
                writer.writerow([filename, label])
        
        print(f"已从 CSV 文件中移除 {len(missing_files)} 个缺失的视频条目。")
    except Exception as e:
        print(f"更新 CSV 文件时出错: {e}")

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description="检查 train_1.csv 中的视频文件是否存在，并可选择删除缺失的条目。")
    parser.add_argument('--remove', action='store_true', help="如果指定，将删除 CSV 文件中不存在的视频条目。")
    args = parser.parse_args()
    
    # 设置路径
    root = Path("/mnt/SSD8T/home/huangwei/projects/FROSTER")
    data_root = root / "data/hmdb51"
    csv_path = root / "zs_label_db/B2N_hmdb/val.csv"
    
    # 检查路径是否存在
    if not data_root.exists():
        print(f"数据集目录不存在: {data_root}")
        return
    if not csv_path.exists():
        print(f"标签文件不存在: {csv_path}")
        return
    
    # 解析 CSV 文件
    print("正在解析 CSV 文件...")
    video_files = parse_train_csv(csv_path)
    print(f"在 CSV 文件中找到 {len(video_files)} 个视频条目。")
    
    # 检查视频文件是否存在
    print("正在检查视频文件是否存在...")
    missing_files = check_videos_exist(data_root, video_files)
    
    # 打印检查结果
    if not missing_files:
        print("所有视频文件均存在于数据集中。")
    else:
        print(f"共有 {len(missing_files)} 个视频文件在数据集中缺失：")
        for filename, label in missing_files:
            print(f"- {filename} : 标签 {label}")
        
        # 根据参数选择是否删除缺失的条目
        if args.remove:
            print("正在从 CSV 文件中移除缺失的条目...")
            remove_missing_from_csv(csv_path, video_files, missing_files)
        else:
            print("如果需要删除缺失的条目，请使用 --remove 参数。")

if __name__ == "__main__":
    main()