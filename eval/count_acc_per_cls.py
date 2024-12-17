import pickle
import torch
import argparse
import json
from collections import defaultdict

'''
使用命令:
python count_acc_per_cls.py --file_path /mnt/SSD8T/home/huangwei/projects/FROSTER/checkpoints/basetraining/B2N_hmdb51_froster/testing/temp.pyth --idx2cls /mnt/SSD8T/home/huangwei/projects/FROSTER/zs_label_db/B2N_hmdb/raw_test_idx2cls.json
'''
def eval_classwise_accuracy(preds, labels, class_names=None):
    """
    统计总体的Top-1准确率和每个类别的准确率。
    
    Args:
        preds (torch.Tensor): 模型预测结果，shape = [N, num_classes]
        labels (torch.Tensor): 真实标签，shape = [N]
        class_names (dict): 类别ID到类别名的映射字典，默认None。
        
    Returns:
        total_accuracy (float): 总体Top-1准确率。
        classwise_accuracy (dict): 每个类别的统计信息。
    """
    # 获取每个样本的预测结果 (Top-1)
    _, predicted = torch.max(preds, dim=1)  # 取每行最大值的索引

    # 统计总体准确率
    total_correct = (predicted == labels).sum().item()
    total_samples = len(labels)
    total_accuracy = total_correct / total_samples * 100.0

    # 统计每个类别的准确率
    class_correct = defaultdict(int)
    class_total = defaultdict(int)
    
    for pred, label in zip(predicted, labels):
        class_total[label.item()] += 1
        if pred.item() == label.item():
            class_correct[label.item()] += 1
    
    # 组装结果
    classwise_accuracy = {}
    for class_id in sorted(class_total.keys()):
        correct = class_correct[class_id]
        total = class_total[class_id]
        accuracy = correct / total * 100.0
        class_name = class_names.get(str(class_id), "") if class_names else ""
        classwise_accuracy[class_id] = {
            "class_name": class_name,
            "accuracy": accuracy,
            "correct": correct,
            "total": total
        }
    return total_accuracy, classwise_accuracy


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate per-class accuracy and overall Top-1 accuracy.")
    parser.add_argument("--file_path", type=str, required=True, help="Path to the saved .pyth file.")
    parser.add_argument("--idx2cls", type=str, default=None, help="Path to the JSON file mapping class indices to class names.")
    args = parser.parse_args()

    # 加载类别名称映射（如果提供了 JSON 文件）
    class_names = {}
    if args.idx2cls:
        try:
            with open(args.idx2cls, "r", encoding="utf-8") as f:
                class_names = json.load(f)
            print("Successfully loaded class index-to-name mapping.")
        except Exception as e:
            print(f"Failed to load class names file: {e}")
            exit()

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

    # 评估准确率
    total_accuracy, classwise_accuracy = eval_classwise_accuracy(all_preds, all_labels, class_names)

    # 输出结果
    print("\nOverall Top-1 Accuracy: {:.2f}%".format(total_accuracy))
    print("\nPer-Class Accuracy:")
    for class_id, stats in classwise_accuracy.items():
        class_id_str = f"class_id: {class_id}"
        class_name_str = f", class_name: {stats['class_name']}" if stats['class_name'] else ""
        print(f"{class_id_str}{class_name_str}, Top-1 acc: {stats['accuracy']:.2f}%, "
              f"{stats['correct']}/{stats['total']}")