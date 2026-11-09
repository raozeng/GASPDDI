import pandas as pd
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import KFold
from model import SMILESTokenizer
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit.Chem import MACCSkeys
from negative_sampling import sample_ryu_negatives
from extract_3d import validate_feature_cache
import warnings
warnings.filterwarnings('ignore')
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')

def build_fp_cache(smiles_list):
    cache = {}
    for smi in set(smiles_list):
        if smi not in cache:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                # 512 (Morgan) + 167 (MACCS Keys) = 679
                cache[smi] = np.zeros(679, dtype=np.float32)
            else:
                morgan = np.array(AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=512), dtype=np.float32)
                maccs = np.array(MACCSkeys.GenMACCSKeys(mol), dtype=np.float32)
                cache[smi] = np.concatenate([morgan, maccs])
    return cache

class RyuDataset(Dataset):
    """
    Ryu 数据集解析器 (对应 drugbank.tab)
    全量正负三元组为 1:1；冷启动筛选后的各折比例可能有所不同。
    """
    def __init__(self, file_path, tokenizer, max_len=0, features_3d_path=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = pd.read_csv(file_path, sep='\t')
        
        self.features_3d = None
        self.features_3d_metadata = None
        
        # 提取数据
        self.smiles1 = self.data['X1'].tolist()
        self.smiles2 = self.data['X2'].tolist()
        # 标签通常从 1 开始，我们需要将其转为 0-85
        self.labels = (self.data['Y'] - 1).tolist()
        unique_smiles = sorted(set(self.smiles1) | set(self.smiles2))
        self.drug_ids = {smiles: i for i, smiles in enumerate(unique_smiles)}
        self.max_smiles_length = max(map(len, unique_smiles), default=0)
        if max_len is not None and (max_len < 0 or (max_len > 0 and self.max_smiles_length > max_len)):
            raise ValueError(f'数据中最长 SMILES 为 {self.max_smiles_length}，超过 max_len={max_len}；使用 --max_len 0，禁止静默截断')
        if features_3d_path:
            payload = torch.load(features_3d_path, map_location='cpu', weights_only=True)
            self.features_3d = validate_feature_cache(payload, self.smiles1 + self.smiles2)
            self.features_3d_metadata = payload['metadata']

        # 负例索引仍与正例一一对应，集合归属由下方划分函数独立决定。
        # 反向等价键去重，防止同一负三元组跟随不同行号跨训练/测试重复。
        self.neg_smiles1, self.neg_smiles2 = sample_ryu_negatives(
            self.smiles1, self.smiles2, self.labels, seed=42
        )
        self.fp_cache = build_fp_cache(list(self.smiles1) + list(self.smiles2) + list(self.neg_smiles1) + list(self.neg_smiles2))
        
    def __len__(self):
        # 长度为真实样本的 2 倍 (一半正样本，一半负样本)
        return 2 * len(self.labels)
        
    def __getitem__(self, idx):
        N = len(self.labels)
        if idx < N:
            s1 = self.smiles1[idx]
            s2 = self.smiles2[idx]
            r = self.labels[idx]
            label = 1.0 # 正样本
        else:
            s1 = self.neg_smiles1[idx - N]
            s2 = self.neg_smiles2[idx - N]
            r = self.labels[idx - N]
            label = 0.0 # 负样本
        
        # 保留完整序列，padding 推迟到当前 batch 汇总时处理。
        ids1 = torch.tensor(self.tokenizer.encode(s1), dtype=torch.long)
        ids2 = torch.tensor(self.tokenizer.encode(s2), dtype=torch.long)
        r_tensor = torch.tensor(r, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float32)
        
        fp_1 = torch.tensor(self.fp_cache[s1])
        fp_2 = torch.tensor(self.fp_cache[s2])
        
        if self.features_3d is not None:
            f3d_1 = self.features_3d.get(s1, torch.zeros(512, dtype=torch.float32))
            f3d_2 = self.features_3d.get(s2, torch.zeros(512, dtype=torch.float32))
        else:
            # 无教师消融也保持相同 batch 格式；训练会跳过对齐损失。
            f3d_1 = torch.zeros(512, dtype=torch.float32)
            f3d_2 = torch.zeros(512, dtype=torch.float32)
        drug1 = torch.tensor(self.drug_ids[s1], dtype=torch.long)
        drug2 = torch.tensor(self.drug_ids[s2], dtype=torch.long)
        valid1 = torch.tensor(self.features_3d is not None and s1 in self.features_3d, dtype=torch.bool)
        valid2 = torch.tensor(self.features_3d is not None and s2 in self.features_3d, dtype=torch.bool)
        return ids1, ids2, f3d_1, f3d_2, fp_1, fp_2, r_tensor, label_tensor, drug1, drug2, valid1, valid2


def collate_ryu_batch(batch):
    """Pad each drug side to its batch maximum; keep all molecular identities."""
    columns = list(zip(*batch))
    ids1 = torch.nn.utils.rnn.pad_sequence(columns[0], batch_first=True, padding_value=0)
    ids2 = torch.nn.utils.rnn.pad_sequence(columns[1], batch_first=True, padding_value=0)
    return (ids1, ids2, *(torch.stack(list(column)) for column in columns[2:]))


class TwosidesDataset(Dataset):
    """
    TWOSIDES 数据集解析器 (对应 twosides_ge_500.csv)
    多标签任务，并补充生成 1:1 的纯负样本对。
    """
    def __init__(self, file_path, tokenizer, max_len=128, features_3d_path=None):
        self.tokenizer = tokenizer
        self.max_len = max_len
        df = pd.read_csv(file_path)
        
        self.features_3d = None
        if features_3d_path:
            import os
            if os.path.exists(features_3d_path):
                self.features_3d = torch.load(features_3d_path)
        
        self.num_classes = int(df['New Y'].max() + 1)
        
        # 按照 (Drug1, Drug2) 聚合 labels
        grouped = df.groupby(['Drug1', 'Drug2'])['New Y'].apply(list).reset_index()
        self.smiles1 = grouped['Drug1'].tolist()
        self.smiles2 = grouped['Drug2'].tolist()
        self.label_lists = grouped['New Y'].tolist()

        # 预先生成等比例的全 0 负样本组合，匹配原论文 DSN-DDI 要求
        import random
        random.seed(42)
        unique_smiles = list(set(self.smiles1 + self.smiles2))
        
        positive_set = set()
        for s1, s2 in zip(self.smiles1, self.smiles2):
            positive_set.add((s1, s2))
            positive_set.add((s2, s1))
            
        self.neg_smiles1 = []
        self.neg_smiles2 = []
        for i in range(len(self.smiles1)):
            while True:
                s1_random = random.choice(unique_smiles)
                s2_random = random.choice(unique_smiles)
                if s1_random != s2_random and (s1_random, s2_random) not in positive_set:
                    break
            self.neg_smiles1.append(s1_random)
            self.neg_smiles2.append(s2_random)
            
        self.fp_cache = build_fp_cache(list(self.smiles1) + list(self.smiles2) + list(self.neg_smiles1) + list(self.neg_smiles2))
        
    def __len__(self):
        return 2 * len(self.label_lists)
        
    def __getitem__(self, idx):
        N = len(self.label_lists)
        if idx < N:
            s1 = self.smiles1[idx]
            s2 = self.smiles2[idx]
            labels = self.label_lists[idx]
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            multi_hot[labels] = 1.0
        else:
            s1 = self.neg_smiles1[idx - N]
            s2 = self.neg_smiles2[idx - N]
            multi_hot = torch.zeros(self.num_classes, dtype=torch.float32)
            
        ids1 = torch.tensor(self.tokenizer.encode(s1, self.max_len), dtype=torch.long)
        ids2 = torch.tensor(self.tokenizer.encode(s2, self.max_len), dtype=torch.long)
        
        fp_1 = torch.tensor(self.fp_cache[s1])
        fp_2 = torch.tensor(self.fp_cache[s2])
        
        if self.features_3d is not None:
            f3d_1 = self.features_3d.get(s1, torch.zeros(512, dtype=torch.float32))
            f3d_2 = self.features_3d.get(s2, torch.zeros(512, dtype=torch.float32))
            return ids1, ids2, f3d_1, f3d_2, fp_1, fp_2, multi_hot
            
        return ids1, ids2, fp_1, fp_2, multi_hot


def get_dataloaders(task, data_path, max_len=0, features_3d_path=None):
    """
    返回完整的数据集对象。具体批次生成在 run.py 中通过 Subset 和 DataLoader 完成。
    """
    tokenizer = SMILESTokenizer()
    
    if task == 'ryu':
        dataset = RyuDataset(data_path, tokenizer, max_len, features_3d_path)
    elif task == 'twosides':
        dataset = TwosidesDataset(data_path, tokenizer, max_len, features_3d_path)
    else:
        raise ValueError("task 必须是 'ryu' 或 'twosides'")
        
    return dataset, tokenizer.vocab_size, (86 if task == 'ryu' else dataset.num_classes)

def generate_kfold_splits(dataset, task_level, n_splits=5, val_ratio=0.25, seed=42, return_refit=False):
    """
    返回每折 (train_idx, val_idx, test_idx)。默认五折约为 60/20/20。
    Task 1 的比例指无序药物对组，正负样本一起分组。
    Task 2/3 的比例指药物数：K 为训练药物，V/U 为验证/测试新药。
    Task 2 使用 K-K / K-V / K-U；Task 3 使用 K-K / V-V / U-U。
    return_refit=True 时返回 (train, val, refit, test)：
    Task 1 重训所有非测试药物对；Task 2/3 重训 (K∪V)-(K∪V)。
    Task 2 的最终测试相应扩展为 (K∪V)-U，Task 3 仍为 U-U。
    """
    if task_level not in ('task1', 'task2', 'task3'):
        raise ValueError("task_level 必须是 task1、task2 或 task3")
    if not 0 < val_ratio < 1:
        raise ValueError("val_ratio 必须在 0 和 1 之间")
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
    splits = []
    all1 = list(dataset.smiles1) + list(getattr(dataset, 'neg_smiles1', []))
    all2 = list(dataset.smiles2) + list(getattr(dataset, 'neg_smiles2', []))

    def inner_split(indices, fold):
        shuffled = np.random.RandomState(seed + fold + 1).permutation(indices)
        n_val = int(np.ceil(len(shuffled) * val_ratio))
        if n_val == 0 or n_val >= len(shuffled):
            raise ValueError("药物或药物对数量不足，无法划分训练集和验证集")
        return shuffled[n_val:], shuffled[:n_val]

    if task_level == 'task1':
        # 正例、负例、不同关系和反向记录共享同一个药物对分组。
        pair_keys = [(min(a, b), max(a, b)) for a, b in zip(all1, all2)]
        groups = sorted(set(pair_keys))
        group_ids = {pair: i for i, pair in enumerate(groups)}
        row_groups = np.array([group_ids[pair] for pair in pair_keys])
        for fold, (remaining, test_groups) in enumerate(kf.split(groups)):
            train_groups, val_groups = inner_split(remaining, fold)
            owner = np.full(len(groups), 2, dtype=np.int8)
            owner[train_groups] = 0
            owner[val_groups] = 1
            row_owner = owner[row_groups]
            parts = [np.flatnonzero(row_owner == i).tolist() for i in range(3)]
            refit_indices = sorted(parts[0] + parts[1])
            # 严格 Known-Known：排除训练中未出现的药物所构成的评估记录。
            known = {drug for i in parts[0] for drug in (all1[i], all2[i])}
            for i in ((1,) if return_refit else (1, 2)):
                parts[i] = [j for j in parts[i] if all1[j] in known and all2[j] in known]
            if return_refit:
                refit_known = {drug for j in refit_indices for drug in (all1[j], all2[j])}
                parts[2] = [j for j in parts[2] if all1[j] in refit_known and all2[j] in refit_known]
                splits.append((parts[0], parts[1], refit_indices, parts[2]))
            else:
                splits.append(tuple(parts))
    else:
        drugs = sorted(set(dataset.smiles1) | set(dataset.smiles2))
        for fold, (remaining, test_drug_ids) in enumerate(kf.split(drugs)):
            train_drug_ids, val_drug_ids = inner_split(remaining, fold)
            owner = {drugs[i]: 0 for i in train_drug_ids}
            owner.update({drugs[i]: 1 for i in val_drug_ids})
            owner.update({drugs[i]: 2 for i in test_drug_ids})
            parts = [[], [], []]
            refit_indices = []
            for i, (a, b) in enumerate(zip(all1, all2)):
                x, y = owner[a], owner[b]
                if x != 2 and y != 2:
                    refit_indices.append(i)
                if x == y == 0:
                    parts[0].append(i)
                elif task_level == 'task2':
                    if (x, y) in ((0, 1), (1, 0)):
                        parts[1].append(i)
                    elif ((x == 2) != (y == 2)) and (return_refit or (x, y) in ((0, 2), (2, 0))):
                        parts[2].append(i)
                elif x == y == 1:
                    parts[1].append(i)
                elif x == y == 2:
                    parts[2].append(i)
            if task_level == 'task2':
                known = {drug for i in parts[0] for drug in (all1[i], all2[i])}
                parts[1] = [j for j in parts[1]
                            if int(all1[j] in known) + int(all2[j] in known) == 1]
                test_known = ({drug for j in refit_indices for drug in (all1[j], all2[j])}
                              if return_refit else known)
                parts[2] = [j for j in parts[2]
                            if int(all1[j] in test_known) + int(all2[j] in test_known) == 1]
            if return_refit:
                splits.append((parts[0], parts[1], refit_indices, parts[2]))
            else:
                splits.append(tuple(parts))

    for fold, parts in enumerate(splits):
        names = ('train', 'val', 'refit', 'test') if return_refit else ('train', 'val', 'test')
        for name, indices in zip(names, parts):
            if not indices:
                raise ValueError(f"Fold {fold+1}: {name} 集合为空，请检查划分参数")
    return splits

if __name__ == "__main__":
    # 简单测试逻辑
    ryu_dataset, vocab, num_classes = get_dataloaders('ryu', r'data\drugbank.tab')
    
    # 测试切分
    task2_splits = generate_kfold_splits(ryu_dataset, 'task2', n_splits=5)
    train_idx, val_idx, test_idx = task2_splits[0]
    print(f"Task 2 Split 0 -> Train pairs: {len(train_idx)}, Val pairs: {len(val_idx)}, Test pairs: {len(test_idx)}")
