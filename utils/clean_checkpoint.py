"""Clean up checkpoint files in the current directory and its subdirectories."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

for p in ROOT.rglob('*'):
    if p.is_file() and p.suffix.lower() in {'.pt', '.pth'}:
        print(f'Deleting: {p}')
        p.unlink()
