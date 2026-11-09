import torch
import torch.nn.functional as F
from tqdm import tqdm

def train_epoch(model, dataloader, optimizer, criterion_ddi, args, device, epoch=0):
    """单轮训练循环 (包含课程学习 OHEM)"""
    model.train()
    total_loss = 0.0
    total_ddi_loss = 0.0
    total_align_loss = 0.0
    
    correct = 0
    total_samples = 0
    
    loop = tqdm(dataloader, desc="Training")
    for batch_data in loop:
        if args.task == 'ryu':
            ids1, ids2, f3d_1, f3d_2, fp_1, fp_2, r, labels, drug1, drug2, valid1, valid2 = batch_data
            r = r.to(device)
            drug1, drug2 = drug1.to(device), drug2.to(device)
        else:
            ids1, ids2, f3d_1, f3d_2, fp_1, fp_2, labels = batch_data
            r = None
            
        ids1, ids2, labels = ids1.to(device), ids2.to(device), labels.to(device)
        f3d_1, f3d_2 = f3d_1.to(device), f3d_2.to(device)
        fp_1, fp_2 = fp_1.to(device), fp_2.to(device)
        
        optimizer.zero_grad()
        
        # 1. 教师模型的高维空间特征提取 (Frozen)
        with torch.no_grad():
            e_3d_1 = f3d_1
            e_3d_2 = f3d_2
        
        # 2. 学生模型 1D 特征
        e_1d_1, e_1d_2, extra = model.encode_pair(ids1, ids2)
        
        # 3. 蒸馏对齐
        if args.lam > 0 and model.aligner is not None:
            # 两侧药物合并后按原始 SMILES 的整数 ID 去重，重复药物不再互为负例。
            loss_align = model.aligner.compute_unique_loss(
                torch.cat((e_1d_1, e_1d_2), dim=0),
                torch.cat((e_3d_1, e_3d_2), dim=0),
                torch.cat((drug1, drug2), dim=0),
                valid_mask=torch.cat((valid1, valid2), dim=0).to(device)
            )
        else:
            loss_align = torch.zeros((), device=device)
        
        # 4 & 5. 融合与任务预测
        logits = model.predictor(e_1d_1, e_1d_2, fp_1, fp_2, **extra) # (B, num_classes)
        
        if args.task == 'ryu':
            binary_logits = logits.gather(1, r.unsqueeze(1)).squeeze(1)
            
            # --- 高阶方案 3：难样本挖掘 (OHEM) 课程学习 ---
            if epoch >= 10 and not getattr(args, 'ablation_no_ohem', False):
                # 高中阶段：动态关闭自动求平均，手动挑选错题
                original_reduction = criterion_ddi.reduction
                criterion_ddi.reduction = 'none'
                
                loss_per_sample = criterion_ddi(binary_logits, labels)
                
                pos_mask = (labels == 1.0)
                neg_mask = (labels == 0.0)
                
                loss_pos = loss_per_sample[pos_mask]
                loss_neg = loss_per_sample[neg_mask]
                
                if len(loss_neg) > 0:
                    # 假阳性错题挖掘：只抽取 Loss 最大的前 50% 负样本 (即长得最像正样本的负样本)
                    k = max(1, len(loss_neg) // 2)
                    hard_neg_loss, _ = torch.topk(loss_neg, k)
                    # 组合最终 Loss (所有的正样本 + 极难分辨的负样本)
                    loss_ddi = (loss_pos.sum() + hard_neg_loss.sum()) / (len(loss_pos) + k)
                else:
                    loss_ddi = loss_pos.mean() if len(loss_pos) > 0 else loss_per_sample.mean()
                    
                criterion_ddi.reduction = original_reduction
            else:
                # 小学阶段：全部样本无差别平均学习
                loss_ddi = criterion_ddi(binary_logits, labels)
            
            with torch.no_grad():
                probs = torch.sigmoid(binary_logits)
                preds = (probs > 0.5).float()
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)
        else:
            if epoch >= 10 and not getattr(args, 'ablation_no_ohem', False):
                original_reduction = criterion_ddi.reduction
                criterion_ddi.reduction = 'none'
                
                # 获取 963 维的全盘 Loss 矩阵
                loss_matrix = criterion_ddi(logits, labels)
                
                # 破局多标签极度不平衡：剥离所有的正样本和负样本
                pos_mask = (labels == 1.0)
                neg_mask = (labels == 0.0)
                
                loss_pos = loss_matrix[pos_mask]
                loss_neg = loss_matrix[neg_mask]
                
                if len(loss_neg) > 0:
                    # 正样本（反应真实发生的）全部保留强制学习；
                    # 负样本（没发生的）只挑出 Loss 最大的前 50% (最容易误判的假阳性) 进行严厉惩罚
                    k = max(1, len(loss_neg) // 2)
                    hard_neg_loss, _ = torch.topk(loss_neg, k)
                    loss_ddi = (loss_pos.sum() + hard_neg_loss.sum()) / (len(loss_pos) + k)
                else:
                    loss_ddi = loss_pos.mean() if len(loss_pos) > 0 else loss_matrix.mean()
                    
                criterion_ddi.reduction = original_reduction
            else:
                loss_ddi = criterion_ddi(logits, labels)
                
            with torch.no_grad():
                probs = torch.sigmoid(logits)
                preds = (probs > 0.5).float()
            correct += (preds == labels).sum().item()
            total_samples += labels.numel()
            
        # 6. 总 Loss
        loss_total = loss_ddi + args.lam * loss_align
        
        loss_total.backward()
        
        # 增加梯度裁剪 (Gradient Clipping)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss_total.item()
        total_ddi_loss += loss_ddi.item()
        total_align_loss += loss_align.item()
        
        current_acc = correct / total_samples if total_samples > 0 else 0.0
        loop.set_postfix(
            Loss=f"{loss_total.item():.4f}", 
            ACC=f"{current_acc:.4f}"
        )
        
    avg_total = total_loss / len(dataloader)
    avg_ddi = total_ddi_loss / len(dataloader)
    avg_align = total_align_loss / len(dataloader)
    train_acc = correct / total_samples if total_samples > 0 else 0.0
    
    # Online training ACC always uses threshold 0.5; thresholds are selected only on validation.
    return avg_total, avg_ddi, avg_align, train_acc, 0.5
