"""Run the existing training entry point with process-local checkpoint protection.

Usage: python checkpoint_safe_launcher.py <the same arguments as run.py>
This does not modify run.py, installed PyTorch, or other running processes.
"""
import copy
import hashlib
import os
from pathlib import Path
import runpy
import sys
import tempfile
import types
import warnings

import torch

_ORIGINAL_SAVE = torch.save
_ORIGINAL_IS_STORAGE = torch.is_storage
_SOURCE = Path(__file__).read_text(encoding='utf-8')
_SOURCE_SHA256 = hashlib.sha256(_SOURCE.encode()).hexdigest()
_LAUNCH_COMMAND = list(sys.argv)


def _check_storage_function():
    current = torch.is_storage
    if isinstance(current, types.CellType):
        if not callable(_ORIGINAL_IS_STORAGE):
            raise RuntimeError('The original torch.is_storage is not callable.')
        warnings.warn(
            'torch.is_storage became a closure cell; restoring the function '
            'captured before training imports. The cause remains undiagnosed.',
            RuntimeWarning, stacklevel=2)
        torch.is_storage = _ORIGINAL_IS_STORAGE
        return True
    if not callable(current):
        raise RuntimeError(
            f'torch.is_storage is {type(current).__name__}, not callable; '
            'refusing to replace the previous checkpoint.')
    return False


def _cpu_snapshot(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu()
    if isinstance(obj, dict):
        result = copy.copy(obj)
        for key, value in obj.items():
            result[key] = _cpu_snapshot(value)
        # copy.copy preserves OrderedDict attributes, including state_dict metadata.
        return result
    if isinstance(obj, list):
        return [_cpu_snapshot(value) for value in obj]
    if isinstance(obj, tuple):
        values = [_cpu_snapshot(value) for value in obj]
        return type(obj)(*values) if hasattr(obj, '_fields') else tuple(values)
    return obj


def safe_save(obj, f, *args, **kwargs):
    """Commit a CPU checkpoint only after the complete temporary write succeeds."""
    _check_storage_function()
    snapshot = _cpu_snapshot(obj)
    if not isinstance(f, (str, bytes, os.PathLike)):
        return _ORIGINAL_SAVE(snapshot, f, *args, **kwargs)
    target = Path(os.fsdecode(f))
    if isinstance(snapshot, dict) and 'model_state_dict' in snapshot:
        snapshot['_checkpoint_save_guard'] = dict(
            version=1, cpu_tensors=True, atomic_replace=True,
            launcher_sha256=_SOURCE_SHA256, launch_command=_LAUNCH_COMMAND)
        archive = target.parent / 'checkpoint_safe_launcher_source.py'
        if archive.exists():
            if archive.read_text(encoding='utf-8') != _SOURCE:
                raise RuntimeError('Checkpoint launcher archive differs from this process.')
        else:
            archive.write_text(_SOURCE, encoding='utf-8')
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=target.name + '.', suffix='.tmp', dir=target.parent)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, 'w+b') as stream:
            for attempt in range(2):
                try:
                    _ORIGINAL_SAVE(snapshot, stream, *args, **kwargs)
                    break
                except TypeError:
                    # Retry only the specific, observed non-callable cell state.
                    if attempt or not _check_storage_function():
                        raise
                    stream.seek(0)
                    stream.truncate()
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        try:
            temporary.unlink(missing_ok=True)
        except OSError as error:
            warnings.warn(f'Could not remove incomplete checkpoint {temporary}: {error}')


def main():
    entrypoint = Path(__file__).resolve().with_name('run.py')
    if not entrypoint.is_file():
        raise FileNotFoundError(entrypoint)
    _check_storage_function()
    torch.save = safe_save
    print('Checkpoint protection: CPU snapshot + atomic save enabled.', flush=True)
    sys.path.insert(0, str(entrypoint.parent))
    sys.argv = [str(entrypoint), *_LAUNCH_COMMAND[1:]]
    runpy.run_path(str(entrypoint), run_name='__main__')


if __name__ == '__main__':
    main()
