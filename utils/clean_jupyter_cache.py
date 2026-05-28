"""Clean up .jupyter_cache directories in the current directory and its subdirectories."""

from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parent.parent

for cache_dir in ROOT.rglob('.jupyter_cache'):
    if cache_dir.is_dir():
        print(f'Deleting: {cache_dir}')
        shutil.rmtree(cache_dir)
