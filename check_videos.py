import os
import random
import cv2
from pathlib import Path
import av

'''
下载了数据集的视频，似乎打不开，执行训练命令的时候报了相关错误
检查视频是否可以正常读取
'''

def check_video_cv2(video_path):
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "Cannot open video with cv2"
        ret, frame = cap.read()
        if not ret:
            return False, "Cannot read first frame"
        cap.release()
        return True, "OK"
    except Exception as e:
        return False, f"CV2 Error: {str(e)}"

def check_video_pyav(video_path):
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            # 只检查第一帧
            break
        container.close()
        return True, "OK"
    except Exception as e:
        return False, f"PyAV Error: {str(e)}"

def main():
    # 数据集路径
    data_root = Path("/mnt/SSD8T/home/huangwei/projects/FROSTER/data/hmdb51")
    
    # 获取所有视频文件
    video_files = []
    for ext in ['.avi', '.mp4', '.mov']:
        video_files.extend(list(data_root.glob(f"**/*{ext}")))
    
    # 随机选择100个视频进行检查
    sample_size = min(100, len(video_files))
    sampled_videos = random.sample(video_files, sample_size)
    
    print(f"Found {len(video_files)} videos, checking {sample_size} random samples...")
    
    # 检查结果统计
    cv2_failed = []
    pyav_failed = []
    
    for i, video_path in enumerate(sampled_videos, 1):
        print(f"\nChecking {i}/{sample_size}: {video_path}")
        
        # 使用CV2检查
        cv2_success, cv2_msg = check_video_cv2(video_path)
        if not cv2_success:
            cv2_failed.append((video_path, cv2_msg))
            print(f"CV2 Check Failed: {cv2_msg}")
        else:
            print("CV2 Check: OK")
            
        # 使用PyAV检查
        pyav_success, pyav_msg = check_video_pyav(video_path)
        if not pyav_success:
            pyav_failed.append((video_path, pyav_msg))
            print(f"PyAV Check Failed: {pyav_msg}")
        else:
            print("PyAV Check: OK")
    
    # 打印统计结果
    print("\n=== Summary ===")
    print(f"Total videos checked: {sample_size}")
    print(f"CV2 failed: {len(cv2_failed)}/{sample_size}")
    print(f"PyAV failed: {len(pyav_failed)}/{sample_size}")
    
    if cv2_failed:
        print("\nCV2 Failed Videos:")
        for path, msg in cv2_failed:
            print(f"- {path}: {msg}")
    
    if pyav_failed:
        print("\nPyAV Failed Videos:")
        for path, msg in pyav_failed:
            print(f"- {path}: {msg}")

if __name__ == "__main__":
    main()