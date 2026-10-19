"""Validate completed experiments, template fidelity, and submission contents."""
import ast
import json
from pathlib import Path
import subprocess
import zipfile

ROOT=Path(__file__).resolve().parents[1]
COURSE=ROOT.parent


def main():
    q1=json.loads((COURSE/'runs/selected_q1.json').read_text())
    chosen=q1['history'][q1['run']]
    q2=json.loads((COURSE/'runs/q2_lr0001/history.json').read_text())
    fcos=json.loads((COURSE/'runs/fcos_full/history.json').read_text())
    detection=json.loads((COURSE/'runs/detection_metrics.json').read_text())
    assert len(chosen)==5 and max(row['map'] for row in chosen)>=.22
    assert len(q2)==50 and max(row['map'] for row in q2)>=.80
    assert len(fcos)==9000 and detection['map']>=20
    source=(ROOT/'report/main.tex').read_text()
    assert source.count('\\includegraphics[')==15
    assert source.count('\\begin{solution}')==5
    for placeholder in ['example-image','filled after','Codex']:
        assert placeholder not in source,placeholder
    with zipfile.ZipFile(ROOT/'report/template/assignment_1.zip') as archive:
        original=archive.read('assignment_1/main.tex').decode()
    expected=original.split('\\begin{document}')[0].replace(
        '\\input{mathabbreviations.tex}','\\input{report/mathabbreviations.tex}')
    actual=(ROOT/'main.tex').read_text().splitlines()[0]+'\n'+source.split('\\begin{document}')[0]
    assert expected==actual,'Template preamble changed'
    survey='\\textbf{Collaboration Survey}'
    assert source[source.index(survey):]==original[original.index(survey):]
    old=subprocess.check_output(['git','show','642c964d384c6ec8da9b37b18c7f88fa1a2feec9:q1_q2_classification/utils.py'],cwd=ROOT,text=True)
    new=(ROOT/'q1_q2_classification/utils.py').read_text()
    for name in ['compute_ap','eval_dataset_map']:
        def function(text):
            return next(node for node in ast.parse(text).body
                        if isinstance(node,ast.FunctionDef) and node.name==name)
        assert ast.dump(function(old))==ast.dump(function(new))
    gt=COURSE/'tools/mAP/input/ground-truth'
    predictions=COURSE/'tools/mAP/input/detection-results'
    assert len(list(gt.glob('*.txt')))==2510
    assert {p.name for p in gt.glob('*.txt')}=={p.name for p in predictions.glob('*.txt')}
    manifest=json.loads((COURSE/'runs/submission_manifest.json').read_text())
    assert manifest['unpacked_bytes']<100_000_000 and manifest['zip_bytes']<100_000_000
    with zipfile.ZipFile(COURSE/'submission.zip') as archive:
        names=archive.namelist()
        assert 'report.pdf' in names
        assert all(name=='report.pdf' or name.startswith('code/') and name.endswith('.py') for name in names)
    print('Targets, 15 figures, five technical answers, template/survey fidelity, protected metrics,')
    print('all 2,510 detection evaluations, and PDF/Python-only upload sizes verified.')


if __name__=='__main__':
    main()
