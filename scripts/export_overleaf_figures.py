#!/usr/bin/env python3
"""Export the referenced Overleaf source and original PDF artwork for the README.

Needs pdflatex and pdftoppm. Diagram formulas and layout come from the captured
Overleaf source, with the manuscript's times/Helvetica font configuration.
"""
from pathlib import Path
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / 'assets/overleaf'
with tempfile.TemporaryDirectory(prefix='orsmt-original-figures-') as folder:
    temp = Path(folder)
    command = ['pdflatex', '-no-shell-escape', '-halt-on-error', '-interaction=nonstopmode',
               '-output-directory=' + str(temp), 'method.tex']
    result = subprocess.run(command, cwd=ART / 'source', capture_output=True, text=True)
    if result.returncode:
        raise RuntimeError(result.stdout[-5000:])
    shutil.copy2(temp / 'method.pdf', ART / 'method.pdf')
    for name in ['method', 'repair-progression', 'domain-case']:
        subprocess.run(['pdftoppm', '-r', '350', '-singlefile', '-png',
                        str(ART / (name + '.pdf')), str(ART / name)], check=True)
print('Exported manuscript method source and original reconstruction/domain figures.')
