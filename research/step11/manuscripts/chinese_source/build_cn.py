"""Rebuild the complete Chinese translation without changing scientific results."""
from pathlib import Path
import subprocess,fitz
ROOT=Path(__file__).resolve().parent
(ROOT/'fonts').mkdir(exist_ok=True)
# Obtain the renderer's CJK fallback for typesetting; no system CJK font needed.
(ROOT/'fonts/chinese.ttf').write_bytes(fitz.Font('china-s').buffer)
for _ in range(2):
    subprocess.run(['xelatex','-interaction=nonstopmode','-halt-on-error','manuscript_cn.tex'],cwd=ROOT,check=True)
print(ROOT/'manuscript_cn.pdf')
