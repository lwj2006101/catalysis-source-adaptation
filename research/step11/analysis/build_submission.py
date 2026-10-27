"""Build submission attachments from audited results, without fitting models."""
from pathlib import Path
from xml.sax.saxutils import escape
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, PageBreak, Table, TableStyle
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_LEFT
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.pdfbase.ttfonts import TTFont

ROOT = Path(__file__).resolve().parents[1]
FACTS = json.loads((ROOT/'reports/step11_measured_facts.json').read_text())
OUT = ROOT/'submission'
OUT.mkdir(exist_ok=True)
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
font_root = Path('/usr/share/fonts/truetype/dejavu')
pdfmetrics.registerFont(TTFont('DejaVu', str(font_root/'DejaVuSans.ttf')))
pdfmetrics.registerFont(TTFont('DejaVuBold', str(font_root/'DejaVuSans-Bold.ttf')))
styles = getSampleStyleSheet()
CN = ParagraphStyle('CN', fontName='STSong-Light', fontSize=10.5, leading=17,
                    wordWrap='CJK', spaceAfter=10)
CNH = ParagraphStyle('CNH', parent=CN, fontSize=15, leading=22, spaceBefore=8, spaceAfter=14)
ENT = ParagraphStyle('ENT', fontName='DejaVu', fontSize=9.8, leading=14.2, spaceAfter=9)
EH = ParagraphStyle('EH', parent=ENT, fontName='DejaVuBold', fontSize=12, leading=16, spaceAfter=14)
CODE = ParagraphStyle('CODE', fontName='DejaVu', fontSize=8.2, leading=12,
                      spaceAfter=10, backColor=colors.HexColor('#eef3f7'), borderPadding=8)


def p(text, style=CN):
    return Paragraph(escape(text).replace('\n','<br/>'), style)


def footer(c, doc):
    c.saveState()
    c.setFont('DejaVu',8)
    c.setFillColor(colors.HexColor('#667788'))
    c.drawString(43,28,'Step11 | 2026-10-11 | completed computational evidence')
    c.drawRightString(doc.pagesize[0]-43,28,str(doc.page))
    c.restoreState()


def pdf(path, flowables, title):
    doc = SimpleDocTemplate(str(path), pagesize=(595.276,841.89),
                            leftMargin=43,rightMargin=43,topMargin=43,bottomMargin=47,
                            title=title,author='',subject='Manuscript and reproducibility package')
    doc.build(flowables,onFirstPage=footer,onLaterPages=footer)


def graphical_abstract():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42})
    fig=plt.figure(figsize=(9.4,4.2),facecolor='white')
    layout=fig.add_axes([0,0,1,1]);layout.set_axis_off()
    layout.text(.035,.93,'What do five target labels learn?',size=19,weight='bold',color='#17354a')
    layout.text(.035,.835,'Separate response level, correspondence and measurement location',size=11,color='#426173')
    for x,w in [(.035,.27),(.335,.28),(.65,.315)]:
        layout.add_patch(FancyBboxPatch((x,.19),w,.57,boxstyle='round,pad=.012',
                                      facecolor='#f3f7fa',edgecolor='#cbd9e3',linewidth=1))
    layout.text(.06,.69,'1  Match the information',weight='bold',size=11,color='#17354a')
    layout.text(.06,.605,'Mean only\nLocal response relation\nHistorical response reuse',size=10.5,va='top',linespacing=1.7)
    layout.text(.06,.325,'Matched supports\n+ support-label permutation',size=9.5,color='#426173',linespacing=1.6)
    layout.text(.36,.69,'2  Fix budget and predictor',weight='bold',size=10.6,color='#17354a')
    layout.text(.36,.595,'Same 5 responses\nSame local RBF predictor\nDifferent acquisition geometry',size=10.3,va='top',linespacing=1.7)
    layout.text(.36,.32,'Query inputs are known;\nquery responses are held out.',size=9.5,color='#426173',linespacing=1.6)
    layout.text(.675,.69,'3  Exclude query catalyst identities',weight='bold',size=9.7,color='#17354a')
    ax=fig.add_axes([.716,.36,.213,.245])
    bars=ax.bar([0,1],[4.921593829794387,4.106935452417888],width=.6,
                color=['#7d95a5','#147b8d'])
    ax.set_ylim(0,6.0);ax.set_xticks([0,1],['Random','Composition V'])
    ax.tick_params(labelsize=8);ax.set_ylabel('Mean MAE (pp)',fontsize=8)
    ax.spines[['top','right']].set_visible(False)
    ax.bar_label(bars,labels=['4.922','4.107'],padding=3,fontsize=9)
    layout.text(.675,.275,'16.6% lower mean error',size=9.5,color='#146b79')
    layout.text(.675,.215,'19/20 partitions improve; one publication',size=8.6,color='#426173')
    layout.text(.035,.088,'Source-disjoint historical controls + within-family catalyst-partition sensitivity',size=10,color='#17354a')
    layout.text(.035,.035,'Labels help through identifiable information and task-dependent geometry.',size=9.7,color='#426173')
    fig.savefig(OUT/'graphical_abstract.pdf',bbox_inches=None)
    fig.savefig(OUT/'graphical_abstract.png',dpi=200,bbox_inches=None)
    plt.close(fig)


def main():
    assert FACTS['historical_audit']['passed'] and FACTS['opened_audit']['passed']
    text=(OUT/'cover_letter.txt').read_text()
    story=[p('Cover letter — submission draft',EH)]
    story.extend(p(x,ENT) for x in text.strip().split('\n\n'))
    pdf(OUT/'cover_letter.pdf',story,'Cover letter: What few target labels learn under publication shift in catalysis')
    graphical_abstract()
    story=[p('第11步：实际成果与投稿安排',CNH),p('2026年10月11日。英文主文12页，完整补充材料45页；科研结果、编辑源文件、预测数组、逐来源统计和前序完整成果均已保留。当前稿件供作者核读和准备投稿，尚未提交到期刊。'),
           p('本轮最有用的新结果',CNH),
           p('同样5个标签、同样的本地RBF预测器，在20次催化剂身份完全分离的划分中，组成V选样的平均MAE为4.107，随机选样为4.922个百分点；均值之比对应约16.6%的误差降低，19/20次划分改善。20次划分来自同一已打开的测量来源，只能说明这个实验家族内的稳定性。'),
           p('方法和对照已经实际执行',CNH),
           p('新增RBF预测算子匹配风险和谱协方差扰动惩罚两个选样对照，使历史比较达到14种策略。完成25个种子/折组合、5种标签预算，并对固定RBF、原冻结混合器、支持均值及零标签模型进行同预算比较。'),
           p('更复杂的谱惩罚未带来稳定优势：25组历史训练选择中23组选择零惩罚、2组选择0.1；完整历史选择零惩罚。历史5标签的惩罚版平均MAE为2.913，名义版为2.828，惩罚反而增加约0.085个百分点。论文保留该结果，主线是标签信息机制和测量位置价值。'),
           p('新增10,948,000条标量预测，全部误差重新汇总；其中2,737,000条固定RBF查询值由独立sklearn计算重新构建，最大差异小于1e-12。一个全新目录的单折复现实际计算54,180条预测，保存的所有数组逐项一致。没有新增全局骨干拟合或湿实验。'),
           PageBreak(),p('如何阅读和复现',CNH),
           p('主文：manuscripts/manuscript_en.pdf。补充材料：manuscripts/supplementary_en.pdf。主文以标签信息、同预算测量位置和查询任务为主线；完整对照、失败结果、来源账本及数学核验保留在SI和运行档案。'),
           p('关键核验记录：reports/step11_measured_facts.json、reports/operator_math_checks.json、reports/portable_smoke_reproduction.json，以及两个实际运行目录下的verification/audit.json。'),
           p('将ZIP完整解压，在step11根目录运行。使用Python 3.12，依赖版本列于requirements.txt，科学计算无需GPU或内网。LaTeX重建另需pdflatex及文内使用的标准宏包。'),
           p('python3 analysis/reproduce.py check',CODE),
           p('以上只读检查确认文件校验和、执行锁、25组历史和20次身份划分完成记录及各项核验结果。'),
           p('python3 analysis/reproduce.py smoke --destination tmp/new_smoke --workers 1',CODE),
           p('以上在新的空目录实际重新计算单折，不能复用完成缓存。完整重跑使用：'),
           p('python3 analysis/reproduce.py compute --destination runs/new_reproduction --workers 4',CODE),
           p('只重建论文图表与PDF：python3 analysis/reproduce.py paper。投稿附件重建：python3 analysis/build_submission.py。更换目录后仍可运行；reference_step10须完整保留。'),
           p('reference_step10内完整包含第10步、第9步及更早成果，原第10步3290个清单文件保持校验和不变。新增公开数据调查及排除原因另有记录；未取得可靠原始表或不允许再分发的新数据未纳入科研证据或交付原始数据。'),
           PageBreak(),p('投稿所需的真实作者信息',CNH),
           p('请在submission/author_metadata.json填入英文姓名、单位、通信邮箱、作者贡献、利益冲突、基金和致谢。需要作者本人确认全体作者同意及独家投稿等声明。这些信息无法由研究文件推断，当前保留空白。'),
           p('python3 analysis/finalize_author_info.py',CODE),
           p('填写后运行该命令生成独立命名的最终署名稿，保留当前作者中性科学主稿。生成后需重新核对署名和排版，并在真实投稿系统提交。AI协助说明已经提供事实性草稿，由作者核对并遵循最终期刊要求。'),
           p('期刊与一区目标',CNH),
           p('Computers & Chemical Engineering作为计算化工方向的候选，已备有适配投稿信。主题和方法范围相符不等于接受保证；其当前JCR分区、中科院分区、具体学科和年份需以投稿人所在单位认可的数据库为准，本包未冒称已核定一区。'),
           p('Digital Discovery在官方页面列为Web of Science ESCI，保留为主题匹配备选。不能仅凭影响因子就判断满足SCI一区目标。所有期刊事实和核查入口在submission/投稿操作说明.txt。'),
           p('一区的主要剩余风险',CNH),
           p('本稿的原创性主要在整合的信息机制框架、对照干预及来源分离的经验证据；岭回归、RBF、V最优设计和谱范数界属于已有数学工具。尚无新独立湿实验家族支持本轮扩展，也未证明新惩罚算法普遍领先。若审稿要求更强原创算法或多实验来源复核，这些是科研升级方向，不能由附件数量替代。'),
           p('完整包已经满足继续核读、复现和准备投稿的文件需求。持久化仓库和真实DOI应由最终存证产生；未提交、未发布仓库、未编造署名或资助。作者完成真实信息和最终期刊选择后，可进入投稿操作。')]
    pdf(ROOT/'manuscripts/第11步_成果与投稿说明.pdf',story,'第11步：实际成果与投稿安排')
    notes='''第11步投稿操作说明（2026-10-11）

当前科学稿和附件已经完成；实际投稿尚未发生。
主文12页，SI45页，均有可编辑LaTeX源和全部图表依赖。推荐先核读主文。

1. 填写真实作者元数据
编辑submission/author_metadata.json。authors格式示例仅为字段结构：
[{"name": "真实英文姓名", "affiliation": "真实英文单位"}]
填写corresponding_email、contributions、conflicts_of_interest、funding、acknowledgements。
基金不存在时使用作者核实的无资助声明。all_authors_approved及
originality_and_exclusive_submission_confirmed由作者本人确认后再置true。
repository_url仅填实际保存成功的公开或审稿访问仓库地址。
运行python3 analysis/finalize_author_info.py生成单独命名的署名稿，再查排版。

2. 候选期刊
Computers & Chemical Engineering：适配计算方法、系统分析和化学工程问题。
官方范围：https://www.sciencedirect.com/journal/computers-and-chemical-engineering
官方作者指南：https://www.sciencedirect.com/journal/computers-and-chemical-engineering/publish/guide-for-authors
投稿信为submission/cover_letter.txt/.pdf；Highlights、图文摘要和可复现附件齐备。
当前指南直接访问曾返回403，未将不可读的要求猜成已完成检查。
具体文件格式、AI披露、上传顺序、匿名审稿或额外声明，在正式投稿入口核查。
当前JCR/CAS分区未取得可引用的官方数据库核定结果；不能据主题适配宣称一区。
官方索引核查入口：https://mjl.clarivate.com/search-results?issn=0098-1354

Digital Discovery：主题适配备选。
官方页面：https://www.rsc.org/publishing/journals/digital-discovery
官方指南：https://www.rsc.org/publishing/publish-with-us/publish-a-journal-article/digital-discovery
官方页面目前明确列出Web of Science ESCI，不能作为已达到SCI一区的证明。
保留单独适配信cover_letter_Digital_Discovery_ESCI.txt。

一区含义可能是JCR学科Q1，也可能是中科院一区，两者不能互换。
最终目标期刊需明确学科、分区系统及年份，并按所在单位认可的口径核对。

3. 准备并上传
署名主文、完整SI、highlights.txt、graphical_abstract、投稿信；按期刊需要提交
编辑源文件与独立图件。代码/数据/预测证据完整留在第11步ZIP中，不能只交PDF。
投稿信中的署名和作者信息待真实信息填写后生成；不要保留中性草稿的提示语。
数据代码的持久化存证需真实产生记录再写入仓库链接和DOI。期刊审稿时应能访问
可复现代码及合规数据。本包未公开发布或替作者发起投稿。

4. 研究声明和范围
20次催化剂划分来自一个已打开实验家族，报告条件性分布；新增预测条数
并不增加独立实验来源数量。两个新对照已实际执行，谱风险惩罚未占优。
本稿为信息机制与测量设计实证研究，不能改写成通用新算法获胜或已达到一区。
现有历史数据来源与许可说明沿用完整前序包。新下载的Nguyen最佳条件数据
受再分发限制并改变任务定义，故未收入交付原始数据，也没有用于计分。
AI协助事实性声明在元数据中；所有催化响应来自已归档实测数据，非LLM生成。
'''
    (OUT/'投稿操作说明.txt').write_text(notes)
    meta=json.loads((OUT/'author_metadata.json').read_text())
    status=dict(scientific_manuscript_complete=True,supplementary_complete=True,
                actual_computations_audited=True,submission_attachments_complete=True,
                article_submitted=False,new_public_repository_published=False,
                journal_quartile_officially_verified=False,
                author_identity_filled=bool(meta['authors']),
                author_declarations_confirmed=bool(meta['all_authors_approved'] and meta['originality_and_exclusive_submission_confirmed']),
                main_candidate='Computers & Chemical Engineering; scope candidate, quartile unverified',
                alternative='Digital Discovery; official ESCI',
                remaining_author_actions=['supply genuine authors and affiliations','confirm contributions, funding and conflict statements','choose and verify target journal','arrange actual repository deposit and final submission'])
    (OUT/'submission_readiness.json').write_text(json.dumps(status,indent=2,ensure_ascii=False)+'\n')
    print('Built cover letter, graphical abstract, Chinese report and submission instructions.')


if __name__=='__main__':main()
