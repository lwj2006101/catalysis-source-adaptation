"""Three-page Chinese account of executed evidence and remaining research gaps."""
from pathlib import Path
import json
from reportlab.pdfgen import canvas
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak,Image,KeepTogether
from reportlab.lib.styles import getSampleStyleSheet,ParagraphStyle
from reportlab.lib import colors
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont
from reportlab.lib.pagesizes import A4

ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
styles=getSampleStyleSheet()
styles.add(ParagraphStyle('CNTitle',fontName='STSong-Light',fontSize=21,leading=29,textColor=colors.HexColor('#173c53'),spaceAfter=15,wordWrap='CJK'))
styles.add(ParagraphStyle('CNHead',fontName='STSong-Light',fontSize=13,leading=19,textColor=colors.HexColor('#173c53'),spaceBefore=10,spaceAfter=7,wordWrap='CJK'))
styles.add(ParagraphStyle('CNBody',fontName='STSong-Light',fontSize=10.5,leading=17,spaceAfter=8,wordWrap='CJK'))
styles.add(ParagraphStyle('CNCell',fontName='STSong-Light',fontSize=9.3,leading=14,wordWrap='CJK'))
styles.add(ParagraphStyle('CNNote',fontName='STSong-Light',fontSize=9,leading=14,textColor=colors.HexColor('#52616b'),spaceAfter=7,wordWrap='CJK'))


def p(text,kind='CNBody'):return Paragraph(text,styles[kind])


def tab(rows,widths):
    rows=[[p(str(x),'CNCell') for x in row] for row in rows]
    t=Table(rows,colWidths=widths,repeatRows=1,hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf2')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),7),
        ('BOTTOMPADDING',(0,0),(-1,-1),7),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7897a8')),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5e1e6'))]))
    return t


def footer(c,d):
    c.saveState();c.setFont('STSong-Light',8.5);c.setFillColor(colors.HexColor('#52616b'))
    c.drawString(42,26,'第9步 | 真实执行结果与论文更新 | 2026-10-11')
    c.drawRightString(A4[0]-42,26,str(d.page));c.restoreState()


def main():
    f=json.loads((ROOT/'reports/step9_measured_facts.json').read_text())
    story=[p('第9步：实验已执行，证据已补上','CNTitle'),
        p('本轮已直接在当前环境执行，不依赖内网服务器。完成25个历史种子/折组合、56次全历史骨干模型拟合，以及1739条新发表实测记录的一次冻结评价。新增核 V 和稳健多视图 V 两个研究方法对照，保留视图门控与机制消融。'),
        p('最有价值的进展','CNHead'),
        p('现在有两项更具体的正面证据：组成信息不仅能进入预测器，也能用来决定测量位置；在新的已发表实测队列上，组成预测器在5个标签下，组成选样相对随机选样降低MAE约12%。这比“标签越多越好”更接近一个可讨论的研究结论。'),
        tab([['数据与预测器','随机选样 MAE','组成选样 MAE','点改善'],
             ['历史 OCM F0','3.705','3.370','9.05%'],
             ['历史 OCM F1','3.092','2.708','12.40%'],
             ['新队列 OCM F0','5.476','5.039','7.99%'],
             ['新队列 OCM F1','4.787','4.213','12.01%']],[155,94,94,93]),
        Spacer(1,10),p('MAE单位均为C₂收率的百分点。F0只用工况预测；F1还用组成描述符。这里统一使用冻结的自适应混合预测器、5个支持标签和随机查询。历史结果属于方法开发；新队列只有一个实验家族，因此不报告跨来源置信区间。','CNNote'),
        p('前序结果保留：标准选样在甲醇STY和组成增强OCM上的家族调整优势仍成立。当前原稿、完整数据、逐行结果与审计记录一并保留，没有把前序失败候选改写成成功方法。'),
        p('新数据的真实性与边界','CNHead'),
        p('作者CADS原始CSV共2928行。按原论文验证阶段核对后，排除1024条对照/出处不明记录、120条缺少完整温度或响应的记录和45条重复条件，保留1739条实测记录、116种催化剂。五个验证阶段属于同一论文和实验家族，不计为五个独立来源。模型和参数在新队列评价前已冻结。'),
        PageBreak(),p('方法原创性：有贡献，但新门控未胜出','CNTitle'),
        p('这次没有把一个名字包装成创新。保守视图门控的历史主要MAE为3.373，固定组成选样为3.370；与五项固定对照的家族调整比较均未支持门控优势。门控在110个主要来源/种子情形中109次仍选组成，说明它基本回到了固定策略。'),
        tab([['新增或关键对照','历史 F0 MAE','判断'],
             ['固定组成 V','3.370','强基线'],
             ['核 V','3.458','没有优于固定组成'],
             ['稳健多视图 V','3.379','没有明确增益'],
             ['保守视图门控','3.373','未证明优势'],
             ['无边际视图门控','3.569','表现更差']],[168,96,172]),
        p('更有解释力的现象','CNHead'),
        p('在新队列中，F0预测器使用工况选样的MAE为4.817，优于组成选样的5.039；但F1预测器使用组成选样为4.213，优于工况选样的4.483。选择器的工作后验方差、真实部署预测器和来源变化之间存在不匹配，不能简单认定“信息越丰富的视图总是更好”。'),
        p('标签为什么有用','CNHead'),
        p('新队列F1的同一组成选样支持集下：全局模型5.869，5个标签的均值预测5.959，本地线性模型4.170，冻结混合模型4.213，历史调参的本地RBF为3.813。标签能支持来源内的输入-响应关系，作用超过一个均值；但这不等于无标签跨任意来源的泛化能力变强。'),
        p('当前更稳妥的原创性定位是：在来源迁移下，分离标签水平、标签对应关系、历史响应结构和采样表征的实证研究。新增门控的经验边际与简单后悔界尚不足以构成“先进新算法”的证据。'),
        PageBreak(),p('一区还差什么，以及如何继续','CNTitle'),
        p('已经补齐的计算证据','CNHead'),
        p('历史比较产生93,150个评价单元、7,948,800个逐行标量预测；新队列产生1,380个单元、7,443,720个标量预测。全部误差重新汇总、全部支持选样顺序重建，另有数值参考与真实目标标签扰动检查。英文完整论文、补充材料、图表、模型、数据出处与日志均已更新。'),
        p('仍然影响一区竞争力的两个实质缺口','CNHead'),
        p('1. 方法优势：当前视图门控没有超过强基线。若以算法创新作为主卖点，需要一种与部署预测器风险更一致的采样机制，并证明它在多来源、不同预算和强对照下稳定增益；不能通过反复更名或挑选有利预算实现。'),
        p('2. 验证范围：新增1739条记录仍是一个论文/实验家族，四项工况和组成百分比也缺失。另一个有完整条件、批次和催化剂标识的实验室，或先冻结采样规则再进行的实测实验，会比继续加相似的文献行更有说服力。'),
        p('我对当前稿件的判断','CNHead'),
        p('主题具备面向一区的研究价值，当前计算证据比前一版完整；稿件更适合定位为有机制分析的催化评测与实验设计研究。现有结果尚不能支持“新算法领先”的叙述，也不能据此保证一区录用。应围绕已经验证的信息分离和同预算收益写主文，把不成功的候选作为边界证据。'),
        p('接手文件与下一步','CNHead'),
        p('先读包根目录的00_第9步进度说明.txt和AGENT_START_HERE.txt。最新论文位于manuscripts，当前真实结果位于runs/history_v1和runs/new_cohorts/nishimura_measured_v1，冻结模型位于runs/full_history_v1。第7步完整原成果保留在reference_step7。'),
        p('新队列已被评价，未来方法修改应把它作为开发证据，不能再次称作未见确认集。ACS SI只有图形测量且缺少完整可用数值表，甲醇候选未取得兼容SI，两者均没有被计成新结果。下一阶段优先补方法机制和不同实验来源。','CNNote')]
    path=ROOT/'manuscripts/第9步_真实结果与一区差距.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,rightMargin=42,leftMargin=42,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print(str(path))


if __name__=='__main__':main()
