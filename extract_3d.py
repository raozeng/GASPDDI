"""Verified Uni-Mol v1 features; strict 3D generation and reusable versioned cache."""
import argparse
import hashlib
import importlib.metadata
import json
import os
import pickle
from pathlib import Path
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.request

import numpy as np
import pandas as pd

WEIGHT_SHA256 = {
    'no_h': 'da27196af09a8c6d089e10b7764b6a716bcc33da227fc118f5b45b0e484585e9',
    'all_h': '7f5f14bb28bf479a0b8f38ebab4bb9d7ae867383f14c4d694d056500fa3446a7',
}


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def download_weights(mode='all_h', weights_dir=None, source='auto', mirror_url='https://hf-mirror.com'):
    """Download official release in verified byte ranges; replace only after SHA256 matches."""
    if mode not in WEIGHT_SHA256:
        raise ValueError('Unknown hydrogen mode')
    directory = Path(weights_dir or Path(__file__).resolve().parent / '权重').resolve()
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f'mol_pre_{mode}_220816.pt'
    if target.exists() and file_sha256(target) == WEIGHT_SHA256[mode]:
        print(f'Weight already verified: {target}', flush=True)
        return str(target)
    if source == 'auto':
        errors = []
        for endpoint in ('mirror', 'github', 'huggingface'):
            try:
                return download_weights(mode, directory, endpoint, mirror_url)
            except (OSError, ValueError, RuntimeError) as exc:
                errors.append(f'{endpoint}: {exc}')
                print(f'Download source {endpoint} failed; trying next source.', flush=True)
        raise RuntimeError('All download sources failed: ' + '; '.join(errors))
    if not mirror_url.startswith('https://'):
        raise ValueError('Mirror URL must use HTTPS')
    endpoints = {
        'github': f'https://github.com/deepmodeling/Uni-Mol/releases/download/v0.1/{target.name}',
        'huggingface': f'https://huggingface.co/dptech/Uni-Mol-Models/resolve/main/{target.name}',
        'mirror': f'{mirror_url.rstrip("/")}/dptech/Uni-Mol-Models/resolve/main/{target.name}',
    }
    if source not in endpoints:
        raise ValueError('Unknown download source')
    url = endpoints[source]
    print(f'Download source: {source}', flush=True)
    probe = urllib.request.Request(url, headers={'Range': 'bytes=0-0'})
    with urllib.request.urlopen(probe, timeout=30) as response:
        content_range = response.headers.get('Content-Range', '')
        if response.status != 206 or not content_range.startswith('bytes 0-0/'):
            raise RuntimeError('Official server did not honor byte ranges')
        total = int(content_range.split('/')[-1])
        resolved_url = response.geturl()
        response.read(1)
    chunk_size = 4 * 1024 * 1024
    ranges = [(i, start, min(start + chunk_size, total)-1)
              for i, start in enumerate(range(0, total, chunk_size))]
    print(f'Downloading {target.name}: {total} bytes in {len(ranges)} ranges', flush=True)
    with tempfile.TemporaryDirectory(prefix='.unimol_download_', dir=directory) as temporary:
        work = Path(temporary)
        def fetch(part):
            index, start, end = part
            last_error = None
            for attempt in range(3):
                try:
                    request = urllib.request.Request(resolved_url, headers={'Range': f'bytes={start}-{end}'})
                    with urllib.request.urlopen(request, timeout=45) as response:
                        if (response.status != 206 or response.headers.get('Content-Range')
                                != f'bytes {start}-{end}/{total}'):
                            raise RuntimeError('Invalid range response')
                        with open(work / str(index), 'wb') as output:
                            remaining = end - start + 1
                            while remaining:
                                block = response.read(min(256 * 1024, remaining))
                                if not block:
                                    raise RuntimeError('Incomplete range response')
                                output.write(block)
                                remaining -= len(block)
                    return index
                except (OSError, RuntimeError) as exc:
                    last_error = exc
            raise RuntimeError(f'Range {index} failed: {last_error}')
        with ThreadPoolExecutor(max_workers=32) as pool:
            futures = [pool.submit(fetch, part) for part in ranges]
            for count, future in enumerate(as_completed(futures), 1):
                future.result()
                if count % 4 == 0 or count == len(ranges):
                    print(f'Downloaded ranges: {count}/{len(ranges)}', flush=True)
        assembled = work / 'verified.pt'
        with open(assembled, 'wb') as output:
            for index, _, _ in ranges:
                with open(work / str(index), 'rb') as source:
                    for block in iter(lambda: source.read(1024 * 1024), b''):
                        output.write(block)
        digest = file_sha256(assembled)
        if assembled.stat().st_size != total or digest != WEIGHT_SHA256[mode]:
            raise ValueError(f'Downloaded weight failed official SHA256 verification: {digest}')
        # Stage in the destination directory so the final file inherits its
        # permissions, rather than a private temporary directory's Windows ACL.
        staging = directory / f'.{target.name}.{work.name}.verified'
        try:
            with open(staging, 'xb') as output, open(assembled, 'rb') as source_file:
                for block in iter(lambda: source_file.read(1024 * 1024), b''):
                    output.write(block)
            os.replace(staging, target)
        finally:
            staging.unlink(missing_ok=True)
    print(f'Official weight verified: {target}; SHA256={digest}', flush=True)
    return str(target)


def default_cache_path(data_path, remove_hs=True):
    source = Path(data_path)
    mode = 'no_h' if remove_hs else 'all_h'
    return source.with_name(f'{source.stem}_3d_features_{mode}.pt')


def expected_metadata(smiles, remove_hs, weights_dir, seed=42, max_atoms=256):
    mode = 'no_h' if remove_hs else 'all_h'
    weight_path = Path(weights_dir) / f'mol_pre_{mode}_220816.pt'
    actual = file_sha256(weight_path)
    if actual != WEIGHT_SHA256[mode]:
        raise ValueError(f'Uni-Mol 权重校验失败: {weight_path}; SHA256={actual}')
    return {
        'schema_version': 2, 'generator': 'ETKDGv3_masked_v2',
        'model': 'unimolv1', 'remove_hs': bool(remove_hs),
        'weight_name': weight_path.name, 'weight_sha256': actual,
        'dictionary_sha256': file_sha256(Path(weights_dir) / 'mol.dict.txt'),
        'smiles_sha256': hashlib.sha256(json.dumps(sorted(set(smiles)), ensure_ascii=True).encode()).hexdigest(),
        'seed': int(seed), 'max_atoms': int(max_atoms), 'dimension': 512,
    }


def _array(value):
    if hasattr(value, 'detach'):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def extract_cls_vectors(result, batch_size):
    """Normalize UniMolRepr APIs without guessing or dropping molecule rows.

    Some releases return {'cls_repr': ...}; others return the CLS list or
    matrix directly. Individual rows may also be CPU/GPU tensors.
    """
    source_type = type(result).__name__
    if isinstance(result, dict):
        if 'cls_repr' not in result:
            raise ValueError(f'Uni-Mol 返回字典缺少 cls_repr；已有键: {list(result)}')
        result = result['cls_repr']
    if not isinstance(result, (list, tuple, np.ndarray)) and not hasattr(result, 'detach'):
        raise ValueError(f'不支持的 Uni-Mol 特征返回类型: {type(result).__name__}（外层 {source_type}）')
    try:
        if isinstance(result, (list, tuple)):
            vectors = np.stack([_array(row) for row in result])
        else:
            vectors = _array(result)
    except (ValueError, TypeError) as exc:
        raise ValueError(f'Uni-Mol 特征行形状不一致（返回类型 {source_type}）') from exc
    if batch_size == 1 and vectors.shape == (512,):
        vectors = vectors.reshape(1, 512)
    if vectors.shape != (batch_size, 512) or not np.issubdtype(vectors.dtype, np.number):
        raise ValueError(f'Uni-Mol 特征应为 ({batch_size}, 512)，实际 {vectors.shape}、{vectors.dtype}（返回类型 {source_type}）')
    with np.errstate(over='ignore', invalid='ignore'):
        vectors = vectors.astype(np.float32, copy=False)
    if not np.isfinite(vectors).all() or not np.all(np.any(vectors != 0, axis=1)):
        raise ValueError('Uni-Mol 特征包含 NaN、Inf、溢出或全零向量')
    return vectors


def validate_feature_cache(payload, smiles, expected=None):
    if not isinstance(payload, dict) or not all(k in payload for k in ('metadata', 'features', 'conformers')):
        raise ValueError('旧版或无来源信息的 3D 缓存，请重新提取')
    metadata = payload['metadata']
    if (not isinstance(metadata, dict) or not isinstance(payload['features'], dict)
            or not isinstance(payload['conformers'], dict)
            or not isinstance(metadata.get('remove_hs'), bool)):
        raise ValueError('3D 缓存结构或氢模式错误')
    if metadata.get('schema_version') not in (1, 2) or metadata.get('model') != 'unimolv1':
        raise ValueError('3D 缓存版本或模型类型错误')
    mode = 'no_h' if metadata.get('remove_hs') else 'all_h'
    if (metadata.get('weight_sha256') != WEIGHT_SHA256[mode]
            or (metadata.get('schema_version'), metadata.get('generator')) not in
               ((1, 'ETKDGv3_strict_v1'), (2, 'ETKDGv3_masked_v2'))
            or metadata.get('dimension') != 512):
        raise ValueError('3D 缓存权重指纹或生成方式错误')
    if expected and any(metadata.get(k) != value for k, value in expected.items()):
        raise ValueError('3D 缓存与当前数据、权重、氢模式或构象参数不一致')
    required = set(smiles)
    features, conformers = payload['features'], payload['conformers']
    valid = {s for s in required if conformers.get(s, {}).get('status') == '3d'}
    if set(features) != valid or set(conformers) != required:
        raise ValueError('3D 缓存分子覆盖不完整或包含多余分子')
    for smi in sorted(required):
        conf = conformers[smi]
        if conf.get('status') == 'unavailable' and metadata.get('schema_version') == 2:
            if not isinstance(conf.get('reason'), str) or not conf['reason']:
                raise ValueError(f'缺失构象没有原因记录: {smi}')
            continue
        vector = _array(features[smi])
        coords = _array(conf['coordinates'])
        atoms = conf['atoms']
        if vector.shape != (512,) or not np.isfinite(vector).all() or not np.any(vector):
            raise ValueError(f'无效 Uni-Mol 特征: {smi}')
        if (conf.get('status') != '3d' or not atoms or coords.shape != (len(atoms), 3)
                or not np.isfinite(coords).all()):
            raise ValueError(f'无效或非三维构象: {smi}')
        if len(atoms) > 1 and not np.any(np.ptp(coords, axis=0) > 1e-7):
            raise ValueError(f'全零或重合坐标: {smi}')
        if metadata.get('remove_hs') and 'H' in atoms:
            raise ValueError(f'去氢模式缓存中仍含氢原子: {smi}')
    return features


def prepare_3d_molecule(smiles):
    """Remove only simple counterions/solvent when exactly one organic unit exists."""
    from rdkit import Chem
    base = Chem.MolFromSmiles(smiles)
    if base is None:
        raise ValueError('无法解析 SMILES')
    fragments = Chem.GetMolFrags(base, asMols=True, sanitizeFrags=True)
    organic = [m for m in fragments if any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    if not organic:
        raise ValueError('纯无机盐或离子，没有可用的有机分子构象')
    if len(organic) != 1:
        raise ValueError('存在多个有机组分，不能自动决定药物主体')
    allowed = {1, 5, 6, 7, 8, 9, 14, 15, 16, 17, 34, 35, 53}
    if any(a.GetAtomicNum() not in allowed for a in organic[0].GetAtoms()):
        raise ValueError('有机主体含金属或当前不支持的元素，保留原结构并跳过蒸馏')
    removed = [m for m in fragments if not any(a.GetAtomicNum() == 6 for a in m.GetAtoms())]
    for mol in removed:
        water = Chem.MolToSmiles(mol) == 'O'
        # Restrict counterions so even disconnected Fe/Pt/Sb are never stripped.
        simple_ion = (mol.GetNumAtoms() == 1
                      and mol.GetAtomWithIdx(0).GetAtomicNum() in {3, 8, 9, 11, 12, 17, 19, 20, 35, 53}
                      and mol.GetAtomWithIdx(0).GetFormalCharge() != 0)
        if not (water or simple_ion):
            raise ValueError('伴随组分不属于水或白名单单原子反离子，保留原结构并跳过蒸馏')
    return organic[0], [Chem.MolToSmiles(m) for m in removed]


def generate_conformer(smiles, remove_hs=True, seed=42, max_atoms=256):
    """Retry genuine RDKit 3D embeddings; never substitute 2D or zero coordinates."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    base, removed = prepare_3d_molecule(smiles)
    mol = None
    used_seed = None
    failure_counts = []
    for attempt in range(6):
        candidate = Chem.AddHs(base)
        params = AllChem.ETKDGv3()
        used_seed = (int(seed) + attempt * 1009) % 2147483647
        params.randomSeed = used_seed
        params.numThreads = 1
        params.maxIterations = 1000
        params.timeout = 30
        params.useRandomCoords = attempt > 0
        params.trackFailures = True
        params.useSmallRingTorsions = attempt >= 3
        params.useMacrocycleTorsions = True
        params.useMacrocycle14config = True
        if attempt >= 3:
            params.maxIterations = 3000
            params.boxSizeMult = 4.0
        if attempt:
            print(f'  Retrying conformer {attempt+1}/6 (chirality preserved; embedding timeout 30s)', flush=True)
        if AllChem.EmbedMolecule(candidate, params) == 0:
            mol = candidate
            break
        failure_counts.append(list(params.GetFailureCounts()))
    if mol is None:
        raise ValueError(f'六次保留手性的 3D 构象生成均失败，RDKit failure counts={failure_counts}')
    optimized = False
    try:
        if AllChem.MMFFHasAllMoleculeParams(mol):
            optimized = AllChem.MMFFOptimizeMolecule(mol, maxIters=500) == 0
        elif AllChem.UFFHasAllMoleculeParams(mol):
            optimized = AllChem.UFFOptimizeMolecule(mol, maxIters=500) == 0
    except (ValueError, RuntimeError):
        pass  # Successful embedding remains valid; optimization status is recorded.
    conf = mol.GetConformer()
    if not conf.Is3D():
        raise ValueError(f'构象不是 RDKit 三维构象: {smiles}')
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]
    coordinates = conf.GetPositions().astype(np.float32)
    if remove_hs:
        # Match Uni-Mol's symbol-based hydrogen removal, including isotopes.
        keep = [i for i, atom in enumerate(atoms) if atom != 'H']
        atoms = [atoms[i] for i in keep]
        coordinates = coordinates[keep]
    count = len(atoms)
    selected = np.arange(count)
    if count > max_atoms:
        selected = np.sort(np.random.RandomState(seed).choice(count, max_atoms, replace=False))
        atoms = [atoms[i] for i in selected]
        coordinates = coordinates[selected]
    if (not atoms or not np.isfinite(coordinates).all()
            or (len(atoms) > 1 and not np.any(np.ptp(coordinates, axis=0) > 1e-7))):
        raise ValueError(f'构象坐标无效: {smiles}')
    return {'atoms': atoms, 'coordinates': coordinates, 'status': '3d',
            'source_smiles': smiles, 'teacher_smiles': Chem.MolToSmiles(base),
            'removed_fragments': removed, 'teacher_scope': 'organic_parent' if removed else 'whole_molecule',
            'attempts_used': attempt + 1, 'embedding_failure_counts': failure_counts,
            'original_atom_count': count, 'selected_atom_indices': selected.tolist(),
            'seed': used_seed, 'optimized': bool(optimized)}


def _save_cache_atomic(cache, path):
    import torch
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, suffix='.pt', delete=False) as stream:
        temporary = Path(stream.name)
    try:
        torch.save(cache, temporary)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build_3d_features(data_path='data/drugbank.tab', out_path=None, remove_hs=True,
                      weights_dir=None, seed=42, max_atoms=256, batch_size=32, force=False):
    import torch
    if batch_size < 1 or not 1 <= max_atoms <= 510:
        raise ValueError('batch_size 必须为正，max_atoms 必须在 1 到 510 之间')
    weights_dir = Path(weights_dir or Path(__file__).resolve().parent / '权重').resolve()
    out_path = Path(out_path or default_cache_path(data_path, remove_hs))
    df = pd.read_csv(data_path, sep='\t')
    unique_smiles = sorted(set(df['X1']) | set(df['X2']))
    expected = expected_metadata(unique_smiles, remove_hs, weights_dir, seed, max_atoms)
    def migrate_verified_v1(cache, required):
        if isinstance(cache, dict) and cache.get('metadata', {}).get('schema_version') == 1:
            old_expected = dict(expected, schema_version=1, generator='ETKDGv3_strict_v1')
            validate_feature_cache(cache, required, old_expected)
            cache['metadata'] = dict(cache['metadata'], **expected)
            cache['metadata']['migrated_from'] = 'ETKDGv3_strict_v1'
        return cache
    if out_path.exists() and not force:
        try:
            cache = torch.load(out_path, map_location='cpu', weights_only=True)
            cache = migrate_verified_v1(cache, unique_smiles)
            validate_feature_cache(cache, unique_smiles, expected)
            _save_cache_atomic(cache, out_path)
            print(f'Using verified Uni-Mol cache: {out_path} ({len(unique_smiles)} drugs)')
            return str(out_path)
        except (ValueError, KeyError, TypeError, RuntimeError, EOFError, OSError, pickle.UnpicklingError) as exc:
            raise ValueError(f'3D 缓存校验失败: {exc}；使用 --refresh_3d 重新提取') from exc

    versions = {}
    for package in ('unimol_tools', 'rdkit', 'torch'):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = 'unknown'
    metadata = dict(expected, versions=versions)
    partial_path = out_path.with_suffix('.partial.pt')
    features, conformers, failures = {}, {}, []
    if force:
        partial_path.unlink(missing_ok=True)
    elif partial_path.exists():
        partial = torch.load(partial_path, map_location='cpu', weights_only=True)
        if not isinstance(partial, dict) or not isinstance(partial.get('features'), dict):
            raise ValueError('断点缓存格式错误；使用 --refresh_3d 重新提取')
        done = set(partial.get('conformers', {}))
        if not done.issubset(unique_smiles):
            raise ValueError('断点缓存包含不属于当前数据的分子；使用 --refresh_3d')
        partial = migrate_verified_v1(partial, done)
        validate_feature_cache(partial, done, expected)
        if partial['metadata'].get('versions') != versions:
            raise ValueError('依赖版本与断点缓存不一致；使用 --refresh_3d 重新提取')
        features, conformers = partial['features'], partial['conformers']
        print(f'Resuming verified 3D cache: {len(features)} / {len(unique_smiles)} drugs', flush=True)

    # Must be set before importing unimol_tools; verify the actual loaded file below.
    os.environ['UNIMOL_WEIGHT_DIR'] = str(weights_dir)
    try:
        from unimol_tools import UniMolRepr
    except ImportError as exc:
        raise ImportError('首次生成 3D 缓存需要安装 unimol_tools、torch 和 rdkit') from exc
    print(f'Initializing UniMol 3D Extractor... ({expected["weight_name"]})')
    unimol = UniMolRepr(data_type='molecule', model_name='unimolv1',
                       remove_hs=remove_hs, batch_size=batch_size,
                       use_cuda=torch.cuda.is_available())
    loaded = getattr(unimol.model, 'pretrain_path', None)
    if (not loaded or Path(loaded).resolve() != weights_dir / expected['weight_name']
            or file_sha256(loaded) != expected['weight_sha256']):
        raise ValueError('unimol_tools 未加载指定的本地权重，请使用支持 UNIMOL_WEIGHT_DIR 的版本')
    print(f'Total unique drugs to process: {len(unique_smiles)}', flush=True)
    for i in range(0, len(unique_smiles), batch_size):
        batch_smiles, batch_conformers = [], []
        for position, smi in enumerate(unique_smiles[i:i+batch_size], i+1):
            if smi in conformers:
                continue
            started = time.monotonic()
            print(f'[3D {position}/{len(unique_smiles)}] Generating conformer; SMILES length={len(smi)}', flush=True)
            try:
                # Stable molecular seed, independent of ordering and batch size.
                mol_seed = (seed + int(hashlib.sha256(smi.encode()).hexdigest()[:8], 16)) % 2147483647
                conf = generate_conformer(smi, remove_hs, mol_seed, max_atoms)
                batch_smiles.append(smi)
                batch_conformers.append(conf)
                print(f'[3D {position}/{len(unique_smiles)}] Done in {time.monotonic()-started:.1f}s', flush=True)
            except (ValueError, RuntimeError) as exc:
                failures.append({'smiles': smi, 'error': str(exc)})
                conformers[smi] = {'status': 'unavailable', 'source_smiles': smi, 'reason': str(exc)}
                print(f'[3D {position}/{len(unique_smiles)}] Failed after {time.monotonic()-started:.1f}s: {exc}', flush=True)
        if not batch_smiles:
            _save_cache_atomic({'metadata': metadata, 'features': features, 'conformers': conformers}, partial_path)
            continue
        # Passing validated coordinates prevents the toolkit from regenerating
        # conformers or silently falling back to 2D for these molecules.
        raw = {'SMILES': batch_smiles, 'atoms': [c['atoms'] for c in batch_conformers],
               'coordinates': [c['coordinates'] for c in batch_conformers]}
        with torch.no_grad():
            result = unimol.get_repr(raw, return_atomic_reprs=False)
        vectors = extract_cls_vectors(result, len(batch_smiles))
        for smi, vector, conf in zip(batch_smiles, vectors, batch_conformers):
            features[smi] = torch.tensor(vector, dtype=torch.float32)
            conformers[smi] = dict(conf, coordinates=torch.tensor(conf['coordinates'], dtype=torch.float32))
        # Both verified vectors and explicit unavailable records survive interruption.
        _save_cache_atomic({'metadata': metadata, 'features': features, 'conformers': conformers}, partial_path)
        print(f'Processed {len(conformers)} / {len(unique_smiles)} drugs... (3D valid={len(features)}; partial cache saved)', flush=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    failures = [{'smiles': s, 'error': c['reason']} for s, c in conformers.items() if c['status'] == 'unavailable']
    report = out_path.with_suffix('.failures.json')
    report.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding='utf-8')
    if not features:
        raise ValueError(f'没有有效教师特征，拒绝启动蒸馏；详情: {report}')
    metadata['coverage'] = {'total_drugs': len(unique_smiles), 'valid_3d': len(features),
                            'unavailable_3d': len(failures),
                            'organic_parent': sum(c.get('teacher_scope') == 'organic_parent' for c in conformers.values())}
    cache = {'metadata': metadata, 'features': features, 'conformers': conformers}
    validate_feature_cache(cache, unique_smiles, expected)
    _save_cache_atomic(cache, out_path)
    print(f'3D coverage: {len(features)}/{len(unique_smiles)}; excluded from distillation: {len(failures)}', flush=True)
    partial_path.unlink(missing_ok=True)
    print(f'Successfully saved verified 3D features to {out_path}!')
    return str(out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Extract verified Uni-Mol v1 molecular features')
    parser.add_argument('--data_path', default='data/drugbank.tab')
    parser.add_argument('--out_path', default=None)
    parser.add_argument('--hydrogens', choices=['no_h', 'all_h'], default='no_h')
    parser.add_argument('--weights_dir', default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--max_atoms', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--force', action='store_true')
    parser.add_argument('--download_weights', action='store_true', help='Download and verify selected weights only')
    parser.add_argument('--download_source', choices=['auto', 'mirror', 'github', 'huggingface'], default='auto')
    parser.add_argument('--mirror_url', default='https://hf-mirror.com')
    cli = parser.parse_args()
    if cli.download_weights:
        download_weights(cli.hydrogens, cli.weights_dir, cli.download_source, cli.mirror_url)
    else:
        build_3d_features(cli.data_path, cli.out_path, cli.hydrogens == 'no_h', cli.weights_dir,
                          cli.seed, cli.max_atoms, cli.batch_size, cli.force)
