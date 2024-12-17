import pickle
import torch
import os
import pandas as pd
from sklearn.metrics import confusion_matrix

# 结果文件路径（在代码中直接定义）
RESULT_FILE_PATH = "/mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/testing/temp.pyth"

# 混淆矩阵导出路径
CONFUSION_MATRIX_CSV = "./hmdb_confusion_matrix_new.csv"

# 定义函数: 评估 Top-k 准确率
def eval_accuracy(preds, labels, topk=(1, 5)):
    maxk = max(topk)
    batch_size = labels.size(0)

    # 获取预测的前 topk 个类别
    _, pred = preds.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(labels.view(1, -1).expand_as(pred))

    # 计算每个 k 的准确率
    results = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        acc_k = correct_k / batch_size * 100.0
        results[f"Top-{k} Accuracy"] = acc_k.item()
    return results

# 定义函数: 导出混淆矩阵为CSV
def export_confusion_matrix(preds, labels, save_path):
    """
    生成并保存混淆矩阵为CSV格式。
    Args:
        preds: 模型预测结果，Tensor，shape = [N, num_classes]
        labels: 真实标签，Tensor，shape = [N]
        save_path: CSV文件保存路径
    """
    # 获取 Top-1 预测结果
    _, pred_classes = preds.topk(1, dim=1, largest=True, sorted=True)
    pred_classes = pred_classes.squeeze(1)  # 去掉多余的维度

    # 计算混淆矩阵
    cm = confusion_matrix(labels.cpu().numpy(), pred_classes.cpu().numpy())
    
    # 转换为 DataFrame 并保存为CSV
    cm_df = pd.DataFrame(cm)
    cm_df.to_csv(save_path, index=True, header=True)
    print(f"Confusion matrix saved to: {save_path}")

# 主代码
if __name__ == "__main__":
    # 加载保存的文件
    try:
        with open(RESULT_FILE_PATH, "rb") as f:
            data = pickle.load(f)
    except Exception as e:
        print(f"Error loading file: {e}")
        exit()

    # 确认数据格式
    if isinstance(data, list) and len(data) == 2:
        all_preds, all_labels = data
        print("Successfully loaded predictions and labels.")
    else:
        print("Unexpected file format. The file does not contain [all_preds, all_labels].")
        exit()

    # 转换数据到 Tensor 格式
    if not isinstance(all_preds, torch.Tensor):
        all_preds = torch.tensor(all_preds)
    if not isinstance(all_labels, torch.Tensor):
        all_labels = torch.tensor(all_labels)

    # 输出数据形状
    print(f"Prediction shape: {all_preds.shape}")
    print(f"Labels shape: {all_labels.shape}")

    # 计算准确率
    accuracy = eval_accuracy(all_preds, all_labels, topk=(1, 5))
    print("\nEvaluation Results:")
    for k, v in accuracy.items():
        print(f"{k}: {v:.2f}%")

    # 导出混淆矩阵为CSV
    export_confusion_matrix(all_preds, all_labels, CONFUSION_MATRIX_CSV)

