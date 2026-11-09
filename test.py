import torch
from tqdm import tqdm
import numpy as np
try:
    from sklearn.metrics import (roc_auc_score, average_precision_score, accuracy_score, 
                                 precision_score, recall_score, f1_score)
except ImportError:
    # 兼容处理
    def roc_auc_score(*args, **kwargs): return 0.0
    def average_precision_score(*args, **kwargs): return 0.0
    def accuracy_score(*args, **kwargs): return 0.0
    def precision_score(*args, **kwargs): return 0.0
    def recall_score(*args, **kwargs): return 0.0
    def f1_score(*args, **kwargs): return 0.0

def test_epoch(model, dataloader, criterion, args, device, threshold=0.5, split='test'):
    """测试/验证只使用原药 SMILES 和指纹，不使用教师特征或近邻替身。"""
    model.eval()
    all_preds = []
    all_probs = []
    all_labels = []
    
    total_loss = 0.0
    correct = 0
    total_samples = 0
    
    if split not in ('train', 'val', 'test'):
        raise ValueError("split 必须是 train、val 或 test")
    loop = tqdm(dataloader, desc={'train': 'Evaluating Train', 'val': 'Validating', 'test': 'Testing'}[split])
    with torch.no_grad():
        for batch_data in loop:
            if args.task == 'ryu':
                ids1, ids2, _, _, fp_1, fp_2, r, labels, _, _, _, _ = batch_data
                r = r.to(device)
            else:
                ids1, ids2, f3d_1, f3d_2, fp_1, fp_2, labels = batch_data
                r = None
                
            labels = labels.to(device)
            
            ids1 = ids1.to(device)
            ids2 = ids2.to(device)
                
            fp_1, fp_2 = fp_1.to(device), fp_2.to(device)
            
            # 直接通过完整模型前向传播
            logits = model(ids1, ids2, fp_1, fp_2) # (B, num_classes)
            
            if args.task == 'ryu':
                # 保持测试 Loss 与训练 Loss 一致
                binary_logits = logits.gather(1, r.unsqueeze(1)).squeeze(1) # (B,)
                loss = criterion(binary_logits, labels)
                probs = torch.sigmoid(binary_logits)
            else:
                loss = criterion(logits, labels)
                probs = torch.sigmoid(logits)
            
            preds = (probs > threshold).float()
            
            batch_correct = (preds == labels).sum().item()
            correct += batch_correct
            total_samples += labels.numel()
            total_loss += loss.item()
            
            current_acc = correct / total_samples if total_samples > 0 else 0.0
            loop.set_postfix(
                Loss=f"{loss.item():.4f}", 
                ACC=f"{current_acc:.4f}"
            )
            
            all_preds.append(preds.cpu().numpy())
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.cpu().numpy())
                
    # 拼接所有批次结果
    all_labels = np.concatenate(all_labels, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    
    # 彻底平铺为 1D 数组，转为标准的纯二分类计算指标，适用于含有全0负样本的 Multi-hot
    all_labels = all_labels.flatten()
    all_probs = all_probs.flatten()
    
    if split == 'val' and getattr(args, 'dynamic_threshold', False):
        best_f1 = 0.0
        best_threshold = 0.5
        # 动态搜索阈值以最大化 F1 分数 (使用 numpy 向量化加速，防止几千万数据算太慢)
        for candidate_threshold in np.arange(0.01, 1.0, 0.01):
            temp_preds = (all_probs > candidate_threshold).astype(float)
            tp = np.sum((temp_preds == 1) & (all_labels == 1))
            fp = np.sum((temp_preds == 1) & (all_labels == 0))
            fn = np.sum((temp_preds == 0) & (all_labels == 1))
            
            if (tp + fp) > 0 and (tp + fn) > 0:
                precision = tp / (tp + fp)
                recall = tp / (tp + fn)
                temp_f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            else:
                temp_f1 = 0.0
                
            if temp_f1 > best_f1:
                best_f1 = temp_f1
                best_threshold = candidate_threshold
                
        all_preds = (all_probs > best_threshold).astype(float)
        threshold = float(best_threshold)
    else:
        all_preds = (all_probs > threshold).astype(float)
    
    
    # 全部统一为二分类 Link Prediction
    # 抛弃 sklearn 的缓慢 acc/f1，改用极速 numpy 位运算 (对 2000 万数据秒出结果)
    acc = np.mean(all_labels == all_preds)
    
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except:
        auc = 0.0
        
    try:
        ap = average_precision_score(all_labels, all_probs)
    except:
        ap = 0.0
        
    tp = np.sum((all_preds == 1) & (all_labels == 1))
    fp = np.sum((all_preds == 1) & (all_labels == 0))
    fn = np.sum((all_preds == 0) & (all_labels == 1))
    if (tp + fp) > 0 and (tp + fn) > 0:
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    else:
        f1 = 0.0
    
    metrics = {
        'ACC': acc,
        'AUC': auc,
        'AP': ap,
        'F1': f1
    }
    metrics['Threshold'] = float(threshold)

    return {key: float(value) for key, value in metrics.items()}
