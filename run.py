import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import numpy as np
import random
import json
import hashlib
import uuid
import sys
import platform
import importlib.metadata
from pathlib import Path
from datetime import datetime, timezone

from dataset import get_dataloaders, generate_kfold_splits, collate_ryu_batch
from model import ChemMambaDDI, FocalLoss
from train import train_epoch
from test import test_epoch
from extract_3d import build_3d_features

def initialize_training(args, vocab_size, num_classes, device):
    """每折创建独立的模型、优化器和调度器。"""
    model = ChemMambaDDI(
        vocab_size=vocab_size, d_model=args.d_model, d_teacher=512,
        num_classes=num_classes,
        task_type=('multiclass' if args.task == 'ryu' else 'multilabel'),
        backbone=args.backbone, fusion_type=args.fusion,
        use_cnn=not args.ablation_no_cnn,
        use_sequence=not getattr(args, 'ablation_no_sequence', False),
        use_fingerprint=not getattr(args, 'ablation_no_fp', False),
        use_distillation=args.lam > 0
    ).to(device)
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith('.bias') or 'embedding' in name.lower():
            no_decay.append(param)
        else:
            decay.append(param)
    optimizer = optim.Adam([
        {'params': decay, 'weight_decay': 1e-5},
        {'params': no_decay, 'weight_decay': 0.0}
    ], lr=args.lr)
    # 按预设最大轮数调度；验证集选择 checkpoint。
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=1e-6
    )
    return model, optimizer, scheduler


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Record this setting: seeds do not promise deterministic native CUDA kernels.
    torch.backends.cudnn.benchmark = False


def sha256_file(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare_run(args, dataset, splits):
    source = Path(__file__).resolve().parent
    config = dict(vars(args))
    config_hash = hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()[:12]
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ') + '_' + config_hash + '_' + uuid.uuid4().hex[:8]
    directory = Path(args.output_dir).resolve() / run_id
    directory.mkdir(parents=True, exist_ok=False)
    split_arrays = {}
    for fold, indices in enumerate(splits, 1):
        for name, values in zip(('train', 'val', 'test'), indices):
            split_arrays[f'fold{fold}_{name}'] = np.asarray(values, dtype=np.int64)
    np.savez_compressed(directory/'splits.npz', **split_arrays)
    np.savez_compressed(directory/'negative_samples.npz',
        drug_a=np.fromiter((dataset.drug_ids[s] for s in dataset.neg_smiles1), dtype=np.int32),
        drug_b=np.fromiter((dataset.drug_ids[s] for s in dataset.neg_smiles2), dtype=np.int32),
        relation=np.asarray(dataset.labels, dtype=np.int32))
    source_text = {name: (source/name).read_text(encoding='utf-8') for name in
                   ('run.py', 'model.py', 'dataset.py', 'train.py', 'test.py', 'extract_3d.py', 'negative_sampling.py')}
    versions = {}
    for package in ('torch', 'numpy', 'pandas', 'rdkit', 'mamba-ssm', 'unimol_tools', 'scikit-learn'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unavailable'
    manifest = dict(run_id=run_id, config=config, config_hash=config_hash, command_line=list(sys.argv),
        data_sha256=sha256_file(args.data_path),
        splits_sha256=sha256_file(directory/'splits.npz'),
        negatives_sha256=sha256_file(directory/'negative_samples.npz'), negative_sampling_seed=42,
        drug_ids=dataset.drug_ids, tokenizer=dataset.tokenizer.char_to_idx,
        teacher_metadata=getattr(dataset, 'features_3d_metadata', None),
        teacher_cache_sha256=sha256_file(args.features_3d_path) if args.lam > 0 else None,
        source_sha256={k: hashlib.sha256(v.encode()).hexdigest() for k,v in source_text.items()},
        source_text=source_text, versions=versions, python=sys.version, platform=platform.platform(),
        cuda=torch.version.cuda, gpu=torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        evaluation_protocol='train_val_select_then_test_without_refit',
        reproducibility='Fixed initialization/data-order seeds; native CUDA kernels may be nondeterministic.',
        metric_protocol={
            'Train_Online_ACC':'Train mode, threshold 0.5, changing weights across batches; descriptive only.',
            'Val_ACC':'Selected inner checkpoint, validation-selected or fixed threshold.',
            'Train_Eval_ACC':'Validation-selected checkpoint, eval mode, same threshold as final test.',
            'ACC':'Test once on the validation-selected checkpoint; no refit or test-based selection.'},
        effective_modules=dict(sequence=not args.ablation_no_sequence, fingerprint=not args.ablation_no_fp,
            fingerprint_type=args.fingerprint_type, cnn=not args.ablation_no_cnn and not args.ablation_no_sequence,
            distillation=args.lam > 0, teacher_shuffle=args.teacher_shuffle,
            ohem=not args.ablation_no_ohem,
            backbone=args.backbone if not args.ablation_no_sequence else None,
            fusion=args.fusion if not args.ablation_no_sequence else 'fingerprint_pair'), folds=[])
    (directory/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    return directory, manifest


def checkpoint_payload(model, optimizer, scheduler, args, manifest, epoch, stage, **extra):
    np_state = np.random.get_state()
    return dict(model_state_dict=model.state_dict(), optimizer_state_dict=optimizer.state_dict(),
        scheduler_state_dict=scheduler.state_dict(), epoch=epoch, stage=stage, args=dict(vars(args)),
        run_id=manifest['run_id'], data_sha256=manifest['data_sha256'],
        splits_sha256=manifest['splits_sha256'], negatives_sha256=manifest['negatives_sha256'],
        source_sha256=manifest['source_sha256'], tokenizer=manifest['tokenizer'],
        teacher_metadata=manifest['teacher_metadata'], effective_modules=manifest['effective_modules'],
        rng_state=dict(python=random.getstate(), numpy=(np_state[0], np_state[1].tolist(), *np_state[2:]),
                       torch=torch.get_rng_state(), cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else []),
        parameter_count=sum(p.numel() for p in model.parameters()),
        **extra)


def configure_teacher(dataset, original_features, indices, shuffle, seed):
    if original_features is None:
        return {}
    dataset.features_3d = dict(original_features)
    if not shuffle:
        return {}
    n = len(dataset.labels)
    seen = set()
    for idx in indices:
        a,b = (dataset.smiles1[idx], dataset.smiles2[idx]) if idx < n else (dataset.neg_smiles1[idx-n], dataset.neg_smiles2[idx-n])
        seen.update((a,b))
    valid = sorted(seen & set(original_features))
    if len(valid) < 2:
        raise ValueError('教师打乱对照至少需要两种有有效 3D 的训练药物')
    order = np.random.default_rng(seed).permutation(len(valid))
    # A random cycle is a derangement: every training drug receives another drug's teacher.
    mapping = {valid[int(a)]: valid[int(b)] for a,b in zip(order, np.roll(order, 1))}
    dataset.features_3d.update({s: original_features[t] for s,t in mapping.items()})
    return mapping


def main(args):
    if args.epochs < 1 or args.task != 'ryu':
        raise ValueError('epochs 必须至少为 1，且当前仅支持 Ryu')
    if args.early_stopping_patience < 0 or not 0 <= args.min_epochs <= args.epochs:
        raise ValueError('early_stopping_patience 不能为负，min_epochs 必须在 0 到 epochs 之间')
    if args.ablation_no_sequence and args.ablation_no_fp:
        raise ValueError('序列和指纹不能同时关闭')
    if args.lam < 0:
        raise ValueError('lam 不能小于零')
    if args.ablation_no_sequence or args.ablation_no_3d:
        args.lam = 0.0
    if args.teacher_shuffle and args.lam == 0:
        raise ValueError('教师打乱需要启用序列和非零蒸馏损失')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    seed_everything(args.seed)
    print('=================================================')
    print(' 3D-Aligned ChemMamba DDI Pipeline Starting...')
    print(f' Device: {device} | Task: {args.task.upper()}')
    print(f' Eval Level: {args.eval_task.upper()} | Splits: {args.n_splits}')
    print(f' Backbone: {args.backbone} | Fusion: {args.fusion}')
    print(f' Sequence: {not args.ablation_no_sequence} | Fingerprint: {not args.ablation_no_fp} ({args.fingerprint_type}) | CNN: {not args.ablation_no_cnn and not args.ablation_no_sequence} | Distillation: {args.lam > 0}')
    print(' Train ACC is online/train-mode at 0.5; compare final Train_Eval_ACC with Test ACC.')
    print('=================================================')
    print(f'\n[1] Loading Data from {args.data_path}...')
    features_path = None
    if args.lam > 0:
        features_path = build_3d_features(data_path=args.data_path, out_path=args.features_3d_path,
            remove_hs=args.unimol_hydrogens == 'no_h', weights_dir=args.unimol_weights_dir,
            seed=args.conformer_seed, max_atoms=args.unimol_max_atoms,
            batch_size=args.unimol_batch_size, force=args.refresh_3d)
        args.features_3d_path = features_path
    full_dataset, vocab_size, num_classes = get_dataloaders(task=args.task, data_path=args.data_path,
        max_len=args.max_len, features_3d_path=features_path)
    if args.fingerprint_type != 'both' and not args.ablation_no_fp:
        for fp in full_dataset.fp_cache.values():
            fp[512:] = 0 if args.fingerprint_type == 'morgan' else fp[512:]
            fp[:512] = 0 if args.fingerprint_type == 'maccs' else fp[:512]
    print(f'Data Loaded! Total Samples: {len(full_dataset)}')
    print(f'Vocab Size: {vocab_size} | Classes/Labels: {num_classes}')
    print(f'SMILES: Full sequences | Max length: {full_dataset.max_smiles_length} | Dynamic padding')
    print(f'\n[2] Generating K-Fold Splits for {args.eval_task}...')
    splits = list(generate_kfold_splits(full_dataset, args.eval_task, args.n_splits,
                        val_ratio=args.val_ratio, seed=args.split_seed, return_refit=False))
    # Convert once so checkpoint metadata is portable with weights_only=True.
    splits = [tuple([int(i) for i in values] for values in fold) for fold in splits]
    directory, manifest = prepare_run(args, full_dataset, splits)
    print(f'Run ID: {manifest["run_id"]}\nArtifacts: {directory}')
    original_features = getattr(full_dataset, 'features_3d', None)
    fold_best_metrics = []
    history = []
    import pandas as pd
    def loader(indices, shuffle=False, seed=0):
        generator = torch.Generator().manual_seed(seed)
        return DataLoader(Subset(full_dataset, indices), batch_size=args.batch_size,
                          shuffle=shuffle, collate_fn=collate_ryu_batch, generator=generator)
    print('\n[3] Starting K-Fold Cross Validation...')
    for fold, (train_idx, val_idx, test_idx) in enumerate(splits):
        selection_seed = args.seed + fold*100
        seed_everything(selection_seed)
        selection_mapping = configure_teacher(full_dataset, original_features, train_idx, args.teacher_shuffle, selection_seed)
        print(f'\n{"="*15} FOLD {fold+1}/{args.n_splits} {"="*15}')
        print(f'Train Pairs: {len(train_idx)} | Val Pairs: {len(val_idx)} | Test Pairs: {len(test_idx)}')
        train_loader, val_loader = loader(train_idx, True, selection_seed), loader(val_idx)
        model, optimizer, scheduler = initialize_training(args, vocab_size, num_classes, device)
        criterion_ddi = FocalLoss(alpha=0.25, gamma=2.0)
        selection_path = directory/f'fold{fold+1}_selection.pth'
        best_metric, epochs_no_improve, patience = float('-inf'), 0, args.early_stopping_patience
        for epoch in range(args.epochs):
            current_lr = optimizer.param_groups[0]['lr']
            avg_total, avg_ddi, avg_align, train_acc, _ = train_epoch(model, train_loader, optimizer, criterion_ddi, args, device, epoch)
            metrics = test_epoch(model, val_loader, criterion_ddi, args, device, threshold=0.5, split='val')
            scheduler.step()
            print(f"  [Epoch {epoch+1}] (LR: {current_lr:.6f}) Train ACC: {train_acc:.4f} | Val ACC: {metrics['ACC']:.4f} | AUC: {metrics['AUC']:.4f} | AP: {metrics['AP']:.4f} | F1: {metrics['F1']:.4f}")
            history.append(dict(Fold=fold+1, Stage='selection', Epoch=epoch+1, LR=current_lr,
                Train_Online_ACC=train_acc, Total_Loss=avg_total, DDI_Loss=avg_ddi, Align_Loss=avg_align,
                **{'Val_'+k: float(v) for k,v in metrics.items()}))
            pd.DataFrame(history).to_csv(directory/'history.csv', index=False)
            if not np.isfinite(metrics['ACC']):
                raise ValueError('验证集 ACC 非有限值')
            if metrics['ACC'] > best_metric:
                best_metric, epochs_no_improve = metrics['ACC'], 0
                torch.save(checkpoint_payload(model, optimizer, scheduler, args, manifest, epoch+1, 'selection',
                    threshold=metrics['Threshold'], threshold_source='inner_validation' if args.dynamic_threshold else 'fixed_0.5',
                    val_metrics=metrics, train_online_acc=train_acc, seed=selection_seed,
                    data_order_rng_state=train_loader.generator.get_state(),
                    train_indices=train_idx, val_indices=val_idx, test_indices=test_idx,
                    teacher_shuffle_mapping=selection_mapping), selection_path)
                print(f'\033[95m  --> New best model for Fold {fold+1}! (Val ACC: {best_metric:.4f})\033[0m')
            else:
                epochs_no_improve += 1
                if patience > 0:
                    print(f'  --> Early stopping counter: {epochs_no_improve} / {patience}')
            if patience > 0 and epoch + 1 >= args.min_epochs and epochs_no_improve >= patience:
                print(f'  --> Early stopping triggered at Epoch {epoch+1}! Evaluating selected model.')
                break
        checkpoint = torch.load(selection_path, map_location='cpu', weights_only=True)
        # One train-set eval at the selected checkpoint, same threshold as its validation metrics.
        model.load_state_dict(checkpoint['model_state_dict'])
        selection_train = test_epoch(model, loader(train_idx), criterion_ddi, args, device,
                                      threshold=checkpoint['threshold'], split='train')
        checkpoint['train_eval_metrics'] = selection_train
        torch.save(checkpoint, selection_path)
        test_metrics = test_epoch(model, loader(test_idx), criterion_ddi, args, device,
                                  threshold=checkpoint['threshold'], split='test')
        row = dict(test_metrics, Train_Online_ACC=checkpoint['train_online_acc'], Train_Eval_ACC=selection_train['ACC'],
            Val_ACC=checkpoint['val_metrics']['ACC'], Best_Epoch=checkpoint['epoch'])
        fold_best_metrics.append(row)
        manifest['folds'].append(dict(fold=fold+1, selection_seed=selection_seed,
            parameter_count=sum(p.numel() for p in model.parameters()),
            selection_checkpoint=selection_path.name,
            selection_checkpoint_sha256=sha256_file(selection_path),
            results={k:float(v) for k,v in row.items()}))
        (directory/'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
        results = pd.DataFrame(fold_best_metrics, index=range(1, len(fold_best_metrics)+1))
        results.to_csv(directory/'results.csv', index_label='Fold')
        print(f"Fold {fold+1} Train Eval ACC: {selection_train['ACC']:.4f} | Test ACC: {test_metrics['ACC']:.4f} | AUC: {test_metrics['AUC']:.4f} | AP: {test_metrics['AP']:.4f} | F1: {test_metrics['F1']:.4f}")
    mean_series, std_series = results.mean(numeric_only=True), results.std(numeric_only=True)
    print(f'\n{args.n_splits}-Fold Cross Validation Final Results')
    for k,v in mean_series.items():
        print(f'  {k}: {v:.4f} ± {std_series[k]:.4f}')
    results.loc['Mean'], results.loc['Std'] = mean_series, std_series
    results['Run_ID'] = manifest['run_id']
    results.to_csv(directory/'results.csv', index_label='Fold')
    print(f'Metrics successfully saved to {directory / "results.csv"}')


def build_parser():
    parser = argparse.ArgumentParser(description="3D-Aligned ChemMamba Run Pipeline")
    parser.add_argument('--task', type=str, required=True, choices=['ryu'], help="Task type (DrugBank/Ryu)")
    parser.add_argument('--data_path', type=str, required=True, help="Path to dataset file")
    parser.add_argument('--features_3d_path', default=None, help="Verified cache path; default depends on hydrogen mode")
    parser.add_argument('--unimol_hydrogens', choices=['no_h', 'all_h'], default='no_h')
    parser.add_argument('--unimol_weights_dir', default=None, help="Local Uni-Mol weights and mol.dict.txt directory")
    parser.add_argument('--conformer_seed', type=int, default=42)
    parser.add_argument('--unimol_max_atoms', type=int, default=256)
    parser.add_argument('--unimol_batch_size', type=int, default=32)
    parser.add_argument('--refresh_3d', action='store_true', help="Regenerate conformers and features before training")
    
    # 实验设置相关参数
    parser.add_argument('--eval_task', type=str, default='task1', choices=['task1', 'task2', 'task3'], help="Cold-start evaluation task level")
    parser.add_argument('--n_splits', type=int, default=5, help="Number of folds for cross validation")
    parser.add_argument('--val_ratio', type=float, default=0.25, help="Validation fraction within the non-test pair/drug groups")
    parser.add_argument('--split_seed', type=int, default=42, help="Random seed for train/validation/test splits")
    parser.add_argument('--seed', type=int, default=42, help="Initialization and training data-order seed")
    parser.add_argument('--output_dir', default='runs', help="Parent directory; every invocation creates a unique run folder")
    
    # 模型架构消融相关参数
    parser.add_argument('--backbone', type=str, default='mamba', choices=['mamba', 'transformer'], help="Student network backbone")
    parser.add_argument('--fusion', type=str, default='concat', choices=['concat', 'cross_attention'], help="DDI Feature fusion strategy")
    parser.add_argument('--ablation_no_3d', action='store_true', help="Disable 3D distillation (lam=0)")
    parser.add_argument('--ablation_no_cnn', action='store_true', help="Disable 1D-CNN local feature extraction")
    parser.add_argument('--ablation_no_sequence', action='store_true', help="Fingerprint-only baseline; disables sequence and its distillation")
    parser.add_argument('--ablation_no_fp', action='store_true', help="Sequence-only model; remove fingerprint module")
    parser.add_argument('--fingerprint_type', choices=['both', 'morgan', 'maccs'], default='both', help="Disable unselected fingerprint bit blocks")
    parser.add_argument('--teacher_shuffle', action='store_true', help="Derange teachers among valid training drugs only; mapping saved per stage")
    parser.add_argument('--ablation_no_ohem', action='store_true', help="Disable hard-negative mining after epoch 10")
    parser.add_argument('--dynamic_threshold', action='store_true', help="Search threshold on validation data only; freeze it for testing")
    
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--early_stopping_patience', type=int, default=15,
                        help="Consecutive epochs without improved Val ACC; 0 disables early stopping")
    parser.add_argument('--min_epochs', type=int, default=0,
                        help="Earliest completed epoch at which early stopping is allowed; counter still accumulates before this")
    parser.add_argument('--batch_size', type=int, default=64, help="Batch size")
    parser.add_argument('--max_len', type=int, default=0, help="0 keeps full SMILES with dynamic padding; positive values are strict length limits, never truncate")
    parser.add_argument('--d_model', type=int, default=256, help="Hidden dimension for student network")
    parser.add_argument('--lam', type=float, default=0.5, help="Lambda weight for alignment distillation loss")
    parser.add_argument('--lr', type=float, default=1e-4, help="Learning rate")
    
    return parser


if __name__ == "__main__":
    main(build_parser().parse_args())
