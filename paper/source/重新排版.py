"""Rebuild the paper from frozen text/tables. No scientific computations are run."""
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import shutil
import subprocess

import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import mm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer

SRC=Path(__file__).resolve().parent
OUT=SRC.parent


def audit_figure():
    plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    fig,axes=plt.subplots(2,2,figsize=(12,7.6),constrained_layout=True)
    items=[('Suvarna 2022: controlled B comparison',[.876,.319],r'$R^2$ (raw STY)',(0,1)),
           ('Xu 2026: archived fixed configuration',[.896,.251],r'$R^2$ (log1p STY)',(0,1)),
           ('Suvarna 2022: same configuration',[57.9,124.0],'Source-equal MAE (mg MeOH / h / gcat)',(0,155)),
           ('Xu 2026: corrected matched five-fold',[69.6198,126.3649],'Source-equal MAE (mg MeOH / h / gcat)',(0,155))]
    for index,(ax,(title,values,ylabel,limits)) in enumerate(zip(axes.flat,items)):
        bars=ax.bar([0,1],values,color=['#8da5b0','#1a7781'],width=.55)
        labels=['Random 80/20','DOI-source 5-fold'] if index==1 else ['Record 5-fold','DOI-source 5-fold']
        ax.set_xticks([0,1],labels)
        ax.set_title(title,pad=16);ax.set_ylabel(ylabel);ax.set_ylim(*limits)
        ax.yaxis.set_major_locator(MaxNLocator(5));ax.set_axisbelow(True);ax.yaxis.grid(alpha=.2)
        for rect,value in zip(bars,values):
            ax.text(rect.get_x()+rect.get_width()/2,value+limits[1]*.022,
                    f'{value:.3f}' if value<1 else f'{value:.1f}',ha='center',va='bottom',fontweight='bold')
    fig.savefig(SRC/'figs/fig7_published_audit.pdf',bbox_inches='tight');plt.close(fig)


def compile_tex(job):
    folder,name,engine=job
    for extension in ['.aux','.out','.toc','.lof','.lot']:
        cached=folder/(name+extension)
        if cached.is_file():cached.unlink()
    last=''
    for pass_number in range(1,5):
        with (folder/f'{name}_compile_pass{pass_number}.txt').open('w') as log:
            run=subprocess.run([engine,'-halt-on-error','-interaction=nonstopmode',name+'.tex'],
                               cwd=folder,stdout=log,stderr=subprocess.STDOUT)
        for extension in ['.aux','.out','.toc','.log','.pdf']:
            written=folder/(name+extension)
            if written.is_file():
                with written.open('rb') as stream:os.fsync(stream.fileno())
        last=(folder/(name+'.log')).read_text(errors='replace')
        if run.returncode:
            raise RuntimeError(str(folder/name)+'\n'+last[-4500:])
        if pass_number>=2 and not re.search(r'Rerun to get|Label\(s\) may have changed|Rerun to get /PageLabels',last):break
    issues=[line for line in last.splitlines() if 'undefined' in line or 'Missing character' in line]
    if issues:raise RuntimeError(str(folder/name)+'\n'+'\n'.join(issues))
    doc=fitz.open(folder/(name+'.pdf'))
    assert len(doc)>0
    return {'file':str((folder/(name+'.pdf')).relative_to(OUT)),'pages':len(doc),
            'passes':pass_number,'overfull_boxes':re.findall(r'Overfull \\hbox \(([^)]+)',last),
            'undefined_references_or_glyphs':issues}


def supplement():
    a=SRC/'supplement_A/supplement_A.pdf';b=SRC/'supplement_B/supplementary_en.pdf'
    counts=[len(fitz.open(p)) for p in [a,b]]
    pdfmetrics.registerFont(TTFont('PaperCJK',str(SRC/'fonts/chinese.ttf')))
    cover=SRC/'supplement_cover.pdf'
    title=ParagraphStyle('Title',fontName='PaperCJK',fontSize=22,leading=34,alignment=TA_CENTER,spaceAfter=20)
    body=ParagraphStyle('Body',fontName='PaperCJK',fontSize=12,leading=21,wordWrap='CJK',spaceAfter=15)
    note=ParagraphStyle('Note',parent=body,fontSize=10,leading=17,textColor=colors.HexColor('#555555'))
    story=[Spacer(1,22*mm),Paragraph('补充材料',title),Paragraph('来源隔离评估揭示文献催化机器学习的泛化缺口：<br/>少标签适配、输入表征与公开研究复现审计',ParagraphStyle('Sub',parent=title,fontSize=15,leading=24)),Spacer(1,12*mm),
           Paragraph(f'A部分　原始协议与审计补充（{counts[0]}页）',body),
           Paragraph('数据构建、特征、预算、原始对照、分配敏感性及公开研究复现细节。页码以A开头。',note),
           Paragraph(f'B部分　扩展方法与完整结果（{counts[1]}页）',body),
           Paragraph('额外来源、强本地与残差/混合对照、全部选样矩阵、学习协方差与视图门控、表征交叉、核及风险消融、身份分离和重复划分。页码、图表及公式以B开头。',note),
           Spacer(1,8*mm),Paragraph('两部分共同对应同一篇主文；不同分析层采用各自归档的匹配对照，统计口径与来源总体保持分开。',note)]
    SimpleDocTemplate(str(cover),pagesize=(210*mm,297*mm),leftMargin=26*mm,rightMargin=26*mm,topMargin=24*mm,bottomMargin=24*mm).build(story)
    combined=fitz.open()
    for path in [cover,a,b]:
        with fitz.open(path) as doc:combined.insert_pdf(doc)
    combined.set_toc([[1,'补充材料总览',1],[1,'Supplement A: Core protocol and audits',2],[1,'Supplement B: Extended analyses',counts[0]+2]])
    combined.set_metadata({'title':'Supplementary information: Source-disjoint catalysis evaluation','author':'Zhang Junyu; Li Wenjie'})
    target=OUT/'02_补充材料/完整补充材料_A与B.pdf'
    combined.save(target,garbage=4,deflate=True)
    return {'file':str(target.relative_to(OUT)),'pages':len(combined),'A_pages':counts[0],'B_pages':counts[1]}


def main():
    audit_figure()
    jobs=[(SRC,'manuscript_cn','xelatex'),(SRC,'manuscript_en','xelatex'),
          (SRC/'supplement_A','supplement_A','xelatex'),(SRC/'supplement_B','supplementary_en','pdflatex')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        reports=list(pool.map(compile_tex,jobs))
    for lang,name in [('cn','论文中文版_原稿逻辑整合修订.pdf'),('en','Manuscript_Original_Logic_Integrated.pdf')]:
        shutil.copyfile(SRC/f'manuscript_{lang}.pdf',OUT/'01_修订论文'/name)
    reports.append(supplement())
    (OUT/'排版检查.json').write_text(json.dumps({'scientific_runs':0,'compiled':reports},ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(reports,ensure_ascii=False,indent=2),flush=True)


if __name__=='__main__':main()
