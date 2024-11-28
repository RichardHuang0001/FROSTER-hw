import os
import cv2
from pathlib import Path
import av

'''
下载了数据集的视频，似乎打不开，执行训练命令的时候报了相关错误
检查视频是否可以正常读取，并删除无法用PyAV打开的视频文件
'''

def check_video_cv2(video_path):
    """
    使用OpenCV检查视频是否可以打开并读取第一帧
    """
    try:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return False, "无法使用cv2打开视频"
        ret, frame = cap.read()
        if not ret:
            return False, "无法读取第一帧"
        cap.release()
        return True, "正常"
    except Exception as e:
        return False, f"CV2 错误: {str(e)}"

def check_video_pyav(video_path):
    """
    使用PyAV检查视频是否可以打开并读取第一帧
    """
    try:
        container = av.open(str(video_path))
        stream = container.streams.video[0]
        for frame in container.decode(stream):
            # 只检查第一帧
            break
        container.close()
        return True, "正常"
    except Exception as e:
        return False, f"PyAV 错误: {str(e)}"

def main():
    # 数据集路径
    data_root = Path("/mnt/SSD8T/home/huangwei/projects/FROSTER/data/hmdb51")
    
    # 获取所有视频文件
    video_files = []
    for ext in ['.avi', '.mp4', '.mov']:
        video_files.extend(list(data_root.glob(f"**/*{ext}")))
    
    print(f"发现 {len(video_files)} 个视频，正在检查所有视频...")
    
    # 检查结果统计
    cv2_failed = []
    pyav_failed = []
    
    for i, video_path in enumerate(video_files, 1):
        print(f"\n正在检查 {i}/{len(video_files)}: {video_path}")
        
        # 使用CV2检查
        cv2_success, cv2_msg = check_video_cv2(video_path)
        if not cv2_success:
            cv2_failed.append((video_path, cv2_msg))
            print(f"CV2 检查失败: {cv2_msg}")
        else:
            print("CV2 检查: 正常")
            
        # 使用PyAV检查
        pyav_success, pyav_msg = check_video_pyav(video_path)
        if not pyav_success:
            pyav_failed.append((video_path, pyav_msg))
            print(f"PyAV 检查失败: {pyav_msg}")
            try:
                os.remove(video_path)
                print(f"已删除损坏的视频文件: {video_path}")
            except Exception as e:
                print(f"删除文件失败: {video_path}, 错误: {str(e)}")
        else:
            print("PyAV 检查: 正常")
    
    # 打印统计结果
    print("\n=== 总结 ===")
    print(f"检查的视频总数: {len(video_files)}")
    print(f"CV2 检查失败: {len(cv2_failed)}/{len(video_files)}")
    print(f"PyAV 检查失败并删除: {len(pyav_failed)}/{len(video_files)}")
    
    if cv2_failed:
        print("\nCV2 检查失败的视频文件:")
        for path, msg in cv2_failed:
            print(f"- {path}: {msg}")
    
    if pyav_failed:
        print("\nPyAV 检查失败并已删除的视频文件:")
        for path, msg in pyav_failed:
            print(f"- {path}: {msg}")

if __name__ == "__main__":
    main()