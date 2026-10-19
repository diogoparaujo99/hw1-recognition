"""Download checksum-verified VOC2007 archives outside the source repository."""
import argparse
import hashlib
from pathlib import Path
import shutil
import tarfile
import urllib.request

FILES = {
    'VOCtrainval_06-Nov-2007.tar': 'c52e279531787c972589f7e41ab4ae64',
    'VOCtest_06-Nov-2007.tar': 'b6e924de25625d8de591ea690078ad9f',
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[2] / 'data')
    args = parser.parse_args()
    args.root.mkdir(parents=True, exist_ok=True)
    for filename, expected in FILES.items():
        path = args.root / filename
        if not path.exists() or hashlib.md5(path.read_bytes()).hexdigest() != expected:
            url = 'https://huggingface.co/datasets/JimmyUnleashed/Pascal_VOC/resolve/main/' + filename
            print('Downloading', url, flush=True)
            urllib.request.urlretrieve(url, path)
        if hashlib.md5(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError(f'Checksum mismatch: {path}')
        with tarfile.open(path) as archive:
            archive.extractall(args.root, filter='data')
    for path in (Path(__file__).resolve().parents[1] / 'data').glob('voc07_*.json'):
        shutil.copy2(path, args.root / path.name)
    root = args.root / 'VOCdevkit' / 'VOC2007'
    for split, count in [('trainval', 5011), ('test', 4952)]:
        ids = (root / 'ImageSets' / 'Main' / f'{split}.txt').read_text().split()
        assert len(ids) == count
        assert all((root / 'JPEGImages' / f'{i}.jpg').is_file() for i in ids)
        assert all((root / 'Annotations' / f'{i}.xml').is_file() for i in ids)
    print('VOC2007 verified: 5,011 trainval and 4,952 test images.')


if __name__ == '__main__':
    main()
