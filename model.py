import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class FocalLoss(nn.Module):
    """
    针对 DDI 链路预测的难样本挖掘 Focal Loss
    能够极大缓解大量 Easy Negatives 带来的梯度淹没问题
    """
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)
        ce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        p_t = p * targets + (1 - p) * (1 - targets)
        loss = ce_loss * ((1 - p_t) ** self.gamma)

        if self.alpha >= 0:
            alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

try:
    from mamba_ssm import Mamba
except ImportError:
    # 兼容处理：如果没有安装 mamba-ssm，提供一个占位符报错，提醒用户安装
    class Mamba(nn.Module):
        def __init__(self, *args, **kwargs):
            super().__init__()
        def forward(self, x):
            raise NotImplementedError("请先成功安装 mamba-ssm (原生或通过 WSL2) 才能使用 Mamba 层。")

class SMILESTokenizer:
    """简易字符级 SMILES 分词器"""
    def __init__(self):
        # 去重且保留顺序；/ 与反斜杠分别编码，PAD/UNK 使用独立索引。
        chars = " ()+-=#%@.0123456789abcdefghilmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ[]/\\"
        chars = list(dict.fromkeys(chars))
        self.char_to_idx = {'<PAD>': 0, '<UNK>': 1}
        self.char_to_idx.update({c: i+2 for i, c in enumerate(chars)})
        self.vocab_size = len(self.char_to_idx)

    def encode(self, smiles, max_len=None):
        if not isinstance(smiles, str) or not smiles:
            raise ValueError('SMILES 必须是非空字符串')
        indices = [self.char_to_idx.get(c, self.char_to_idx['<UNK>']) for c in smiles]
        if max_len is not None and max_len != 0:
            if max_len < 1 or len(indices) > max_len:
                raise ValueError(f'SMILES 长度 {len(indices)} 超过 max_len={max_len}，拒绝截断；使用 --max_len 0 保留完整序列')
            indices = indices + [self.char_to_idx['<PAD>']] * (max_len - len(indices))
        return indices


def masked_mean_pool(sequence, mask):
    """Only real tokens contribute, including at the boundaries of the CNN."""
    masked = sequence.masked_fill(~mask.unsqueeze(-1), 0.0)
    count = mask.sum(dim=1, keepdim=True).clamp_min(1).to(sequence.dtype)
    return masked.sum(dim=1) / count


class MambaStudent(nn.Module):
    """模态2: 1D 序列极速学生 (Serial Mamba -> CNN)"""
    def __init__(self, vocab_size, d_model=256, d_state=16, d_conv=4, expand=2, use_cnn=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.use_cnn = use_cnn
        
        self.mamba = Mamba(
            d_model=d_model,
            d_state=d_state,
            d_conv=d_conv,
            expand=expand,
        )
        
        if self.use_cnn:
            # 单尺度 1D-CNN: 恢复跑分最高的 kernel_size=7
            self.local_cnn = nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=7, padding=3)
            
    def forward(self, input_ids, return_tokens=False):
        # input_ids: (B, L)
        mask = input_ids.ne(0)
        x = self.embedding(input_ids) # (B, L, D)
        
        # 1. 第一阶段：全局骨架提取 (Mamba)
        out = self.mamba(x) # (B, L, D)
        # Mamba 的 padding 位置可能保留状态，卷积前清零以免污染末尾真实 token。
        out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        
        if self.use_cnn:
            # 2. 第二阶段：局部细节增强
            out_cnn = out.transpose(1, 2) # (B, D, L)
            out = F.relu(self.local_cnn(out_cnn))
            out = out.transpose(1, 2) # (B, L, D)
            
        pooled = masked_mean_pool(out, mask)
        return (pooled, out, mask) if return_tokens else pooled

class TransformerStudent(nn.Module):
    """消融实验: 模态2 替换 Mamba 的 Transformer 基线模型"""
    def __init__(self, vocab_size, d_model=256, nhead=4, num_layers=2, dim_feedforward=512, use_cnn=True):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model, padding_idx=0)
        self.use_cnn = use_cnn
        if use_cnn:
            self.local_cnn = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
    def forward(self, input_ids, return_tokens=False):
        # input_ids: (B, L)
        mask = input_ids.ne(0)
        x = self.embedding(input_ids) # (B, L, D)
        positions = torch.arange(x.size(1), device=x.device, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, x.size(2), 2, device=x.device, dtype=torch.float32)
                                * (-math.log(10000.0) / x.size(2)))
        pe = torch.zeros(x.size(1), x.size(2), device=x.device, dtype=x.dtype)
        pe[:, 0::2] = torch.sin(positions * frequencies).to(x.dtype)
        pe[:, 1::2] = torch.cos(positions * frequencies[:pe[:, 1::2].size(1)]).to(x.dtype)
        x = x * math.sqrt(x.size(2)) + pe.unsqueeze(0)
        out = self.transformer(x, src_key_padding_mask=~mask) # (B, L, D)
        out = out.masked_fill(~mask.unsqueeze(-1), 0.0)
        if self.use_cnn:
            out = F.relu(self.local_cnn(out.transpose(1, 2))).transpose(1, 2)
        pooled = masked_mean_pool(out, mask)
        return (pooled, out, mask) if return_tokens else pooled

class UniMolTeacher(nn.Module):
    """模态1: 3D 空间高维老师 (占位与适配器)"""
    def __init__(self, d_teacher=768):
        super().__init__()
        # 在这里集成真实的 Uni-Mol+ 预训练模型
        # 由于此处没有 Uni-Mol 环境，我们假定数据加载时已经提取好了 3D 特征，
        # 或者这只是一个冻结的特征提取网络。
        self.d_teacher = d_teacher
    
    def forward(self, graph_data):
        # 实际应调用: return self.unimol_model(graph_data)
        # 此处返回随机特征作为示例 (B, d_teacher)
        batch_size = graph_data.size(0)
        return torch.randn(batch_size, self.d_teacher, device=graph_data.device)

class CrossModalAligner(nn.Module):
    """跨模态知识蒸馏对齐模块"""
    def __init__(self, d_student=256, d_teacher=768, d_proj=256):
        super().__init__()
        self.mlp_1d = nn.Sequential(
            nn.Linear(d_student, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj)
        )
        self.mlp_3d = nn.Sequential(
            nn.Linear(d_teacher, d_proj),
            nn.ReLU(),
            nn.Linear(d_proj, d_proj)
        )
        
    def forward(self, e_1d, e_3d):
        z_1d = self.mlp_1d(e_1d)
        z_3d = self.mlp_3d(e_3d)
        return z_1d, z_3d
        
    def compute_infonce_loss(self, z_1d, z_3d, tau=0.1):
        """计算 InfoNCE 损失 L_align"""
        z_1d = F.normalize(z_1d, dim=1)
        z_3d = F.normalize(z_3d, dim=1)
        
        # 计算余弦相似度矩阵
        logits = torch.matmul(z_1d, z_3d.T) / tau
        labels = torch.arange(logits.size(0), device=logits.device)
        return F.cross_entropy(logits, labels)

    def compute_unique_loss(self, student, teacher, drug_ids, tau=0.1, valid_mask=None):
        """One contrastive entry per raw-SMILES identity across both drug sides.

        Average repeated student views so all occurrences receive gradients.
        Teacher vectors are frozen and must be finite, nonzero cached features.
        """
        if (student.ndim != 2 or teacher.ndim != 2 or drug_ids.ndim != 1
                or student.size(0) != teacher.size(0) or student.size(0) != drug_ids.size(0)):
            raise ValueError('对齐特征和药物 ID 数量不一致')
        if drug_ids.dtype not in (torch.int32, torch.int64):
            raise ValueError('药物 ID 必须为整数，不能使用截断 token 作为身份')
        if valid_mask is not None:
            if valid_mask.dtype != torch.bool or valid_mask.shape != drug_ids.shape:
                raise ValueError('3D 有效标记必须为与药物 ID 等长的 bool 张量')
            student, teacher, drug_ids = student[valid_mask], teacher[valid_mask], drug_ids[valid_mask]
        if not torch.isfinite(teacher).all() or (teacher.abs().sum(dim=1) == 0).any():
            raise ValueError('教师特征包含无效值或全零向量，拒绝计算对齐损失')
        if student.size(0) == 0:
            return student.sum() * 0.0
        unique_ids, inverse = torch.unique(drug_ids, sorted=True, return_inverse=True)
        count = torch.bincount(inverse, minlength=unique_ids.numel()).unsqueeze(1)
        student_sum = student.new_zeros((unique_ids.numel(), student.size(1)))
        student_sum.index_add_(0, inverse, student)
        frozen_teacher = teacher.detach()
        teacher_sum = frozen_teacher.new_zeros((unique_ids.numel(), teacher.size(1)))
        teacher_sum.index_add_(0, inverse, frozen_teacher)
        unique_student = student_sum / count.to(student.dtype)
        unique_teacher = teacher_sum / count.to(teacher.dtype)
        if unique_ids.numel() < 2:
            return unique_student.sum() * 0.0
        z_1d, z_3d = self(unique_student, unique_teacher)
        return self.compute_infonce_loss(z_1d, z_3d, tau)

class DDIPredictor(nn.Module):
    """极速双药 DDI 融合与预测 (二分类 Link Prediction)"""
    def __init__(self, d_model=256, num_classes=86, task_type='binary', fusion_type='concat', d_fp=679,
                 use_sequence=True, use_fingerprint=True):
        super().__init__()
        self.task_type = task_type
        self.fusion_type = fusion_type
        self.d_fp = d_fp
        self.use_sequence = use_sequence
        self.use_fingerprint = use_fingerprint
        if not (use_sequence or use_fingerprint):
            raise ValueError('序列与指纹不能同时关闭')
        
        self.fp_proj = nn.Sequential(
            nn.Linear(d_fp, 256),
            nn.ReLU(),
            nn.Dropout(0.2)
        ) if use_fingerprint else None
        self.mamba_dropout = nn.Dropout(0.5)
        
        fusion_dim = (4 * d_model if use_sequence else 0) + (4 * 256 if use_fingerprint else 0)
        if self.fusion_type == 'cross_attention' and use_sequence:
            # Attention over actual SMILES character positions, not pooled singleton tokens.
            self.attn = nn.MultiheadAttention(embed_dim=d_model, num_heads=4, batch_first=True)
        if self.fusion_type not in ('concat', 'cross_attention'):
            raise ValueError("fusion_type 必须是 'concat' 或 'cross_attention'")
            
        self.mlp = nn.Sequential(
            nn.Linear(fusion_dim, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, ea, eb, fp_a=None, fp_b=None, tokens_a=None, tokens_b=None, mask_a=None, mask_b=None):
        features = []
        if self.use_sequence:
            if self.fusion_type == 'cross_attention':
                if tokens_a is None or tokens_b is None or mask_a is None or mask_b is None:
                    raise ValueError('cross_attention 需要未池化的序列及 padding mask')
                a2b, _ = self.attn(tokens_a, tokens_b, tokens_b,
                                  key_padding_mask=~mask_b, need_weights=False)
                b2a, _ = self.attn(tokens_b, tokens_a, tokens_a,
                                  key_padding_mask=~mask_a, need_weights=False)
                ea = ea + masked_mean_pool(a2b, mask_a)
                eb = eb + masked_mean_pool(b2a, mask_b)
            ea, eb = self.mamba_dropout(ea), self.mamba_dropout(eb)
            features.extend([ea, eb, ea * eb, torch.abs(ea - eb)])
        if self.use_fingerprint:
            if fp_a is None or fp_b is None:
                raise ValueError('指纹模块已启用，但没有提供指纹')
            fpa_d, fpb_d = self.fp_proj(fp_a), self.fp_proj(fp_b)
            features.extend([fpa_d, fpb_d, fpa_d * fpb_d, torch.abs(fpa_d - fpb_d)])
        return self.mlp(torch.cat(features, dim=-1))

class ChemMambaDDI(nn.Module):
    """完整的 3D-Aligned ChemMamba 框架"""
    def __init__(self, vocab_size, d_model=256, d_teacher=768, num_classes=86, task_type='multiclass', backbone='mamba', fusion_type='concat', use_cnn=True,
                 use_sequence=True, use_fingerprint=True, use_distillation=True):
        super().__init__()
        
        # 架构消融：选择学生网络的主干
        self.use_sequence = use_sequence
        self.d_model = d_model
        self.fusion_type = fusion_type
        if not use_sequence:
            self.student = None
        elif backbone == 'mamba':
            self.student = MambaStudent(vocab_size=vocab_size, d_model=d_model, use_cnn=use_cnn)
        elif backbone == 'transformer':
            self.student = TransformerStudent(vocab_size=vocab_size, d_model=d_model, use_cnn=use_cnn)
        else:
            raise ValueError("backbone 必须是 'mamba' 或 'transformer'")
            
        self.aligner = CrossModalAligner(d_student=d_model, d_teacher=d_teacher, d_proj=d_model) if use_sequence and use_distillation else None
        
        # 融合机制消融：选择预测器的融合层
        self.predictor = DDIPredictor(d_model=d_model, num_classes=num_classes, task_type=task_type, fusion_type=fusion_type,
                                      use_sequence=use_sequence, use_fingerprint=use_fingerprint)
        self.task_type = task_type

    def encode_pair(self, smiles_a, smiles_b):
        if not self.use_sequence:
            empty = self.predictor.mlp[0].weight.new_zeros((smiles_a.size(0), self.d_model))
            return empty, empty, {}
        if self.fusion_type == 'cross_attention':
            ea, ta, ma = self.student(smiles_a, return_tokens=True)
            eb, tb, mb = self.student(smiles_b, return_tokens=True)
            return ea, eb, dict(tokens_a=ta, tokens_b=tb, mask_a=ma, mask_b=mb)
        return self.student(smiles_a), self.student(smiles_b), {}

    def forward(self, smiles_a, smiles_b, fp_a=None, fp_b=None):
        """推理阶段的前向传播"""
        ea, eb, extra = self.encode_pair(smiles_a, smiles_b)
        logits = self.predictor(ea, eb, fp_a, fp_b, **extra)
        return logits
