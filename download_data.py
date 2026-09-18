"""Download using local Kaggle credentials, or extract an existing ZIP."""
import argparse
import subprocess
import zipfile
from pathlib import Path
from common import ROOT, local_path

DATASET = 'sriramr/fruits-fresh-and-rotten-for-classification'

def extract_zip(archive, destination):
    destination = Path(destination).resolve()
    with zipfile.ZipFile(archive) as zf:
        for item in zf.infolist():
            target = (destination / item.filename).resolve()
            if not target.is_relative_to(destination):
                raise ValueError('ZIP chứa đường dẫn không hợp lệ.')
        zf.extractall(destination)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--zip', help='ZIP đã tải bằng trình duyệt; không cần Kaggle API.')
    parser.add_argument('--output', default='fruit_dataset')
    args = parser.parse_args()
    dest = local_path(args.output)
    dest.mkdir(parents=True, exist_ok=True)
    if args.zip:
        archive = local_path(args.zip)
    else:
        # venv Scripts/bin chứa kaggle cùng Python đang chạy.
        import sys
        executable = Path(sys.executable).parent / ('kaggle.exe' if sys.platform == 'win32' else 'kaggle')
        subprocess.run([str(executable), 'datasets', 'download', '-d', DATASET, '-p', str(dest)], check=True)
        archive = dest / 'fruits-fresh-and-rotten-for-classification.zip'
    extract_zip(archive, dest)
    candidates = sorted({p.parent for p in dest.rglob('train') if p.is_dir() and (p.parent/'test').is_dir()})
    print('Đã giải nén. Đường dẫn dataset có train/test:')
    for p in candidates:
        print(p)
    if not candidates:
        print('Chưa thấy cặp train/test. Kiểm tra cấu trúc ZIP và truyền --data cho train.py.')

if __name__ == '__main__':
    main()
