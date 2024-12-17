import pickle
import torch

# 指定保存的文件路径
file_path = "/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/testing/temp.pyth"

# 加载文件
try:
    with open(file_path, "rb") as f:
        data = pickle.load(f)  # 使用pickle加载数据
except Exception as e:
    print(f"Error loading file: {e}")
    exit()

# 解析加载的数据
if isinstance(data, list) and len(data) == 2:
    all_preds, all_labels = data
    print("Successfully loaded predictions and labels.")
    print(f"Number of predictions: {len(all_preds)}")
    print(f"Number of labels: {len(all_labels)}")
else:
    print("Unexpected file format. The file does not contain [all_preds, all_labels].")
    exit()

# 检查并打印部分数据示例
if isinstance(all_preds, torch.Tensor):
    print("\nSample Predictions:", all_preds[:5].tolist())
else:
    print("\nSample Predictions:", all_preds[:5] if len(all_preds) > 0 else "No predictions found.")

if isinstance(all_labels, torch.Tensor):
    print("Sample Labels:", all_labels[:5].tolist())
else:
    print("Sample Labels:", all_labels[:5] if len(all_labels) > 0 else "No labels found.")
