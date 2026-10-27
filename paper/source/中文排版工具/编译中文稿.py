from pathlib import Path
import subprocess, shutil
src=Path(__file__).resolve().parents[1]
for _ in range(2):
    subprocess.run(["xelatex","-halt-on-error","-interaction=nonstopmode","manuscript_cn.tex"],cwd=src,check=True)
target=src.parent/"01_修订论文/论文中文版_原稿逻辑整合修订.pdf"
shutil.copyfile(src/"manuscript_cn.pdf",target)
print(target)
