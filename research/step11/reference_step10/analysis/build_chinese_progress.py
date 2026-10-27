"""Three-page Chinese report grounded in Step10 executed results."""
from pathlib import Path
import json
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.cidfonts import UnicodeCIDFont

ROOT=Path(__file__).resolve().parents[1]
pdfmetrics.registerFont(UnicodeCIDFont('STSong-Light'))
styles=getSampleStyleSheet()
for name,size,lead,color in [('CNTitle',20,28,'#173c53'),('CNHead',13,19,'#173c53'),
                             ('CNBody',10.5,17,'#202b35'),('CNCell',9.5,14,'#202b35'),
                             ('CNNote',9.3,15,'#52616b')]:
    styles.add(ParagraphStyle(name,fontName='STSong-Light',fontSize=size,leading=lead,
        textColor=colors.HexColor(color),wordWrap='CJK',spaceAfter=8,
        spaceBefore=8 if name=='CNHead' else 0,keepWithNext=name in ('CNHead','CNTitle')))


def p(text,style='CNBody'): return Paragraph(text,styles[style])


def table(rows,widths):
    t=Table([[p(str(x),'CNCell') for x in row] for row in rows],colWidths=widths,repeatRows=1)
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e4edf2')),
        ('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),
        ('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),6),
        ('BOTTOMPADDING',(0,0),(-1,-1),6),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#7897a8')),
        ('LINEBELOW',(0,1),(-1,-1),.3,colors.HexColor('#d5e1e6'))]))
    return t


def footer(c,d):
    c.saveState(); c.setFont('STSong-Light',8.5); c.setFillColor(colors.HexColor('#52616b'))
    c.drawString(42,26,'第10步 | 预测器匹配选样与真实执行结果 | 2026-10-11')
    c.drawRightString(A4[0]-42,26,str(d.page)); c.restoreState()


def main():
    f=json.loads((ROOT/'reports/step10_measured_facts.json').read_text())
    v=f['F1_k5_mae']; h=v['history_random']; e=v['opened_random']; c=v['opened_catalyst']
    rows=[['选样策略','历史随机查询','已打开队列随机','催化剂留出']]
    for name,key in [('随机','random'),('组成 V','composition_vopt'),('有限截距核','finite_gp_v'),
                     ('中心化平方风险','centered_mse'),('中心化绝对风险','centered_mae'),('多核平均绝对风险','hedged_mae')]:
        rows.append([name]+[f'{q[key]["matched_kernel"]:.3f}' for q in [h,e,c]])
    story=[p('第10步：补了方法，也验证了边界','CNTitle'),
      p('这轮已在当前环境完成实跑。新增5个选样机制对照和同调参的各向同性核预测器；完成25组历史种子/折比较，随后冻结历史参数，在此前已评价的1739条实测记录上增加催化剂身份完全分离的压力测试。完整前序成果一并保留。'),
      p('新增的正面证据','CNHead'),
      p('把29种催化剂的435条记录作为查询，剩余87种催化剂的1304条记录作为支持池，两边催化剂标识没有交集。相同本地RBF预测器、5个标签，匹配核选样的MAE为3.689，随机选样为4.818，点改善23.4%；组成V为4.680。这说明测量位置在没有目标催化剂支持标签的查询中仍可能有用。'),
      table(rows,[149,115,115,132]),
      Spacer(1,8),p('表内均为同一个历史选择的F1扩展核预测器，5个标签，MAE单位为C2收率的百分点。历史含22个合格来源；后两列是同一已打开论文的不同查询划分，不能算两个独立数据集。','CNNote'),
      p('必须一同报告的反向结果','CNHead'),
      p('匹配核选样在普通随机查询上变差：扩展核的中心化平方风险MAE为5.460，随机为4.848，组成V为4.164。新增测试展示了任务依赖，尚不能推出一个普遍领先的选样算法。原第7步两项经家族调整的选样优势和第9步首次冻结队列的12.0%点改善仍完整保留。'),
      PageBreak(),p('为什么标签有用，方法新在哪里','CNTitle'),
      p('标签的作用不是一个笼统的“增强泛化”','CNHead'),
      p('目标来源的标签可以估计响应水平，也可以建立该来源内的输入—响应关系。原稿中同标签均值、对应关系打乱、本地模型、残差模型及混合模型的对比，分别检验这些作用。增益属于给定目标来源少量测量后的条件预测，不表示无标签模型对任何新实验室都自动更准。'),
      p('本轮针对选样与预测器的错配','CNHead'),
      p('用实际部署的自由截距核回归权重计算工作误差风险，依次比较有限截距、中心化平方风险、中心化绝对风险，以及历史误差加权的多核版本。预测器在同一特征视图下跨所有选样策略保持相同参数，避免把模型变化误写成选样收益。另用相同历史随机标签调参的9候选核，对照90候选的扩展核。'),
      p('原创性判断：目前是实验与机制贡献','CNHead'),
      p('自由截距核回归、通用克里金、核的加和/乘积以及高斯绝对风险都已有理论来源。本轮贡献是把这些因素拆开，针对来源内校准与不同查询任务进行公平、可复现的检验。不能把标准公式或新名称当成原创定理，也不能把工作风险当成已校准的催化测量误差。'),
      p('主候选没有胜出','CNHead'),
      p('预先锁定的多核平均绝对风险主候选，历史MAE为3.030，组成V为2.863，中心化绝对风险为2.845。五项家族调整比较均未支持主候选优势。中心化平方风险达到2.792，但有限截距对照也为2.790；催化剂留出中三项策略的5点支持集相同。因此点改善不能归因于中心化或绝对风险单一机制。'),
      p('扩展核也没有形成独立领先证据','CNHead'),
      p('25组历史调参都选择了块结构核；同组成V支持下，扩展核2.863，同调参各向同性核2.897，差距很小。已打开队列中扩展核4.164，本地RBF3.813；催化剂留出中扩展核4.199，本地RBF3.689。选择频率不能代替实际预测优势。'),
      PageBreak(),p('当前完成度与一区差距','CNTitle'),
      p('已经完成并可交付的内容','CNHead'),
      p(f'完整历史网格产生{f["historical_cells"]:,}个评价单元，三种队列查询产生{f["application_cells"]:,}个单元，共{f["total_scalar_predictions"]:,}个标量查询预测。所有标量误差重新汇总，16,836条支持序列检查通过；825,840个核预测值用独立距离/核实现与scikit-learn重建。新增零次全局骨干拟合；本轮确实执行了核调参、选样与预测。'),
      p('数值验证包含自由截距解、常数平移、90种核的半正定性、独立联合协方差计算及50,000次高斯模拟。模拟仅检验数学恒等式，未计为催化测量。论文、SI、4张新科学图、全部逐行数组、冻结参数、审计日志和接手提示词均已更新。'),
      p('影响一区竞争力的核心缺口','CNHead'),
      p('1. 如果以新算法为主卖点，还缺跨来源、跨任务、跨预算且超过强本地核与标准选样的稳定增益。目前多核主候选不支持这个叙述。可继续研究历史与目标来源的风险错配、来源级选择依据或已有稳健主动学习方法，不必受本轮候选限制。'),
      p('2. 新压力测试增强了任务覆盖，但仍来自同一实验家族。另一个条件、组成、催化剂与批次标识完整的独立实验来源，或预先冻结规则后的实测实验，比继续增加相似的文献行更有说服力。这轮没有新增第二个独立来源或湿实验。'),
      p('3. 稿件仍需收束主线。建议围绕“标签提供什么信息、同预算测哪里、为什么任务改变会改变选样收益”组织主文。现有正面结果可以支撑有方法分析的催化研究，失效候选作为解释边界；不要把执行规模本身当成科学创新。'),
      p('我对目前水平的判断','CNHead'),
      p('主题与证据有面向一区的潜力，计算对照和可复现性已较完整；方法原创性与独立实验验证仍是主要短板。当前适合以机制化评测和实验设计认识定位，尚不能凭结果保证一区录用。'),
      p('接手顺序','CNHead'),
      p('先读00_第10步进度说明.txt、01_现有成果与路径索引.txt和AGENT_START_HERE.txt。当前论文在manuscripts，真实结果在runs/history_v1和runs/applications/nishimura_opened_v1，历史冻结参数在runs/full_history_v1。完整前序成果在reference_step9。可能有没想到的地方，需要你自己检查一下。','CNNote')]
    path=ROOT/'manuscripts/第10步_真实结果与一区差距.pdf'
    SimpleDocTemplate(str(path),pagesize=A4,leftMargin=42,rightMargin=42,topMargin=40,bottomMargin=42).build(story,onFirstPage=footer,onLaterPages=footer)
    print(path)


if __name__=='__main__':main()
