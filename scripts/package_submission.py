"""Assemble only the report and assignment Python sources; enforce 100 MB."""
from pathlib import Path
import hashlib
import json
import shutil
import zipfile

ROOT=Path(__file__).resolve().parents[1]
COURSE=ROOT.parent
SUBMISSION=COURSE/'submission'
LIMIT=100_000_000


def main():
    SUBMISSION.mkdir(exist_ok=True)
    report=COURSE/'build/main.pdf'
    if not report.is_file():
        raise FileNotFoundError('Compile the completed report before packaging.')
    if report.stat().st_mtime < (ROOT/'report/main.tex').stat().st_mtime:
        raise RuntimeError('The PDF is older than the report source; recompile first.')
    shutil.copy2(report,SUBMISSION/'report.pdf')
    code=SUBMISSION/'code'
    if code.exists():
        shutil.rmtree(code)
    for directory in ['q1_q2_classification','detection','scripts']:
        for source in (ROOT/directory).rglob('*.py'):
            if 'mAP' in source.relative_to(ROOT).parts:
                continue
            destination=code/source.relative_to(ROOT)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(source,destination)
    files=[SUBMISSION/'report.pdf']+sorted(code.rglob('*.py'))
    total=sum(path.stat().st_size for path in files)
    if total>=LIMIT:
        raise RuntimeError(f'Unpacked submission exceeds limit: {total} bytes')
    archive=COURSE/'submission.zip'
    with zipfile.ZipFile(archive,'w',zipfile.ZIP_DEFLATED) as z:
        for path in files:
            z.write(path,path.relative_to(SUBMISSION))
    if archive.stat().st_size>=LIMIT:
        raise RuntimeError('Submission ZIP exceeds 100 MB')
    manifest=dict(unpacked_bytes=total,zip_bytes=archive.stat().st_size,
                  files={str(p.relative_to(SUBMISSION)):hashlib.sha256(p.read_bytes()).hexdigest()
                         for p in files})
    (COURSE/'runs/submission_manifest.json').write_text(json.dumps(manifest,indent=2))
    print(f'{len(files)} files; unpacked {total:,} bytes; ZIP {archive.stat().st_size:,} bytes')


if __name__=='__main__':
    main()
