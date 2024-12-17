import pickle
import torch
import argparse
import os
import matplotlib.pyplot as plt

'''
使用命令:
- 显示图像:
python eval_acc.py --file_path --file_path /mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/testing/temp.pyth --show

- 保存图像:
python eval_acc.py --file_path /mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/testing/temp.pyth  --saveimg ./

- 同时显示和保存:
python eval_acc.py --file_path /path/to/temp.pyth --show --saveimg /path/to/save
'''

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

# 定义函数: 可视化预测结果并保存图像
def visualize_predictions(preds, labels, save_path=None, num_samples=5, show=False):
    """
    可视化置信度分布图并保存结果
    Args:
        preds: 模型预测结果，Tensor，shape = [N, num_classes]
        labels: 真实标签，Tensor，shape = [N]
        save_path: 图像保存的路径 (str)
        num_samples: 可视化的样本数量
        show: 是否显示图像
    """
    num_samples = min(num_samples, len(preds))
    os.makedirs(save_path, exist_ok=True) if save_path else None  # 创建保存目录

    for i in range(num_samples):
        plt.figure(figsize=(10, 6))
        plt.bar(range(preds.size(1)), preds[i].tolist())
        plt.title(f"Sample {i+1}: True Label = {labels[i].item()}")
        plt.xlabel("Class Index")
        plt.ylabel("Confidence")

        # 保存图像
        if save_path:
            img_path = os.path.join(save_path, f"sample_{i+1}.png")
            plt.savefig(img_path)
            print(f"Saved visualization to: {img_path}")

        if show:
            plt.show()
        plt.close()

# 主代码
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate accuracy and optionally visualize predictions.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the saved .pyth file.")
    parser.add_argument("--show", action="store_true", help="Enable visualization of predictions.")
    parser.add_argument("--saveimg", type=str, help="Path to save visualized prediction images.")
    args = parser.parse_args()

    # 加载保存的文件
    try:
        with open(args.file_path, "rb") as f:
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

    # 可视化预测结果
    if args.show or args.saveimg:
        print("\nVisualizing sample predictions...")
        visualize_predictions(all_preds, all_labels, save_path=args.saveimg, show=args.show, num_samples=5)
