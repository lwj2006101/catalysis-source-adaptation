from pathlib import Path
from collections import Counter
import re,json
import fitz
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties, findfont
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT=Path(__file__).resolve().parents[2]
SRC=ROOT/'03_可编辑稿件'
OUT=SRC/'figs_cn'
OUT.mkdir(exist_ok=True)
fontpath=SRC/'fonts/chinese.ttf'
font=fitz.Font(fontfile=str(fontpath))
fp=FontProperties(fname=str(fontpath))
D={
'MAE difference / observed random-design MAE (%)':'误差差值 / 随机选样误差（%）',
'Methanol conversion':'甲醇合成：CO₂转化率',
'Methanol selectivity':'甲醇选择性',
'Methanol STY':'甲醇时空收率',
'OCM operating (F0)':'OCM：仅工况（F0）',
'OCM composition (F1)':'OCM：工况加组成（F1）',
'Five labels | unchanged frozen blend':'5条标签；组合预测模型配置相同',
'Five-comparison adjusted source-group intervals':'区间已作5项比较校正',
'F0: operating predictor':'F0：仅工况预测器',
'F1: composition predictor':'F1：工况加组成预测器',
'MAE (C2-yield percentage points)':'C₂收率MAE（百分点）',
'Historical development (22 sources)':'历史文献（22个来源）',
'Operating V':'工况V',
'Composition V':'组成V',
'New published application (one family)':'作者实测数据（1个实验家族）',
'MAE minus random (pp)':'相对随机选样的MAE差（百分点）',
'Pool V':'候选池V',
'Expanded-kernel risk':'扩展核风险',
'RBF-matched risk':'RBF匹配风险',
'A  Twenty catalyst-disjoint partitions':'A  20次催化剂编号分离划分',
'Random':'随机',
'Comp. V':'组成V',
'Expanded':'扩展核',
'risk':'风险',
'MAE (pp)':'MAE（百分点）',
'B  Query definition changes rankings':'B  查询任务改变方法排序',
'Ordinary query':'普通查询',
'Initial catalyst holdout':'初始编号留出',
'record':'记录随机',
'source-group':'来源分组',
'5-fold':'五折',
'pooled R2':'合并R²',
'CO2 conversion (pp)':'CO₂转化率（百分点）',
'MeOH selectivity (pp)':'甲醇选择性（百分点）',
'STY (mg h−1 g−1':'甲醇时空收率',
'cat)':'',
'OCM C2 yield (pp)':'OCM C₂收率（百分点）',
'labels k':'标签数 k',
'source-equal MAE':'来源等权MAE',
'MeOH conversion (pp)':'甲醇转化率（百分点）',
'base':'历史模型',
'train_mean':'训练均值',
'target_mean':'新来源均值',
'offset_mean':'均值偏移',
'affine':'仿射校正',
'eb':'经验贝叶斯',
'target_ridge':'局部岭',
'OCM C2 yield, F0 (pp)':'OCM F0（百分点）',
'OCM C2 yield, F1 (pp)':'OCM F1（百分点）',
'MAE(F1) − MAE(F0)  (pp)':'MAE差（F1减F0，百分点）',
'OCM, identical rows/sources/folds':'OCM：相同记录、来源与划分',
'F0 operating':'F0：仅工况',
'F1 + composition':'F1：加入组成',
'mean within-source collision':'',
'fraction (main cohort)':'',
'Same-input different-outcome records':'同输入、异结果的记录比例',
'Dalign = mean permuted MAE − real MAE':'D：打乱后MAE减正确配对MAE',
'(>0: gain depends on true label-input alignment)':'大于0表示依赖真实输入与结果的对应',
'T allocation':'测试集分配编号',
'source-equal MAE (k=5)':'来源等权MAE（k=5）',
'OCM F0 (pp)':'OCM F0（百分点）',
'OCM F1 (pp)':'OCM F1（百分点）',
'STY (mg h−1 g−1)':'甲醇时空收率',
'Record 5-fold':'记录随机五折',
'DOI-source 5-fold':'DOI来源五折',
'R2 (raw STY)':'R²（原始STY）',
'Suvarna 2022: controlled B comparison':'Suvarna 2022：受控比较',
'Random 80/20':'随机80/20',
'R2 (log1p STY)':'R²（对数STY）',
'Xu 2026: archived fixed configuration':'Xu 2026：归档固定配置',
'Source-equal MAE (mg MeOH / h / gcat)':'来源等权MAE（mg甲醇 / h / g催化剂）',
'Suvarna 2022: same configuration':'Suvarna 2022：相同配置',
'Xu 2026: corrected matched five-fold':'Xu 2026：后续匹配五折',
}
for k in [5,10]:
 D[f'affine k={k}']=f'仿射 k={k}'
 D[f'target_ridge k={k}']=f'局部岭 k={k}'

missing_characters=sorted({c for label in D.values() for c in label if not c.isspace() and not font.has_glyph(ord(c))})
assert not missing_characters,missing_characters

def lines(page):
 for block in page.get_text('dict')['blocks']:
  for line in block.get('lines',[]):
   yield line,''.join(s['text'] for s in line['spans'])

def nums(page):
 return Counter(t for l,t in lines(page) if re.fullmatch(r'[−\-+]?\d+(?:\.\d+)?',t.strip()))

def drawing_signature(d):
 return (d['type'],str(d['items']),str(d.get('color')),str(d.get('fill')),round(d.get('width') or 0,6))

reports=[]
for source in sorted((SRC/'figs').glob('*.pdf')):
 if source.name=='fig1_design.pdf':continue
 doc=fitz.open(source);page=doc[0]
 old_numbers=nums(page)
 text_rects=[fitz.Rect(l['bbox'])+(-1,-1,1,1) for l,t in lines(page)]
 old_paths=Counter(drawing_signature(d) for d in page.get_drawings()
                   if not any(r.contains(d['rect']) for r in text_rects))
 pending=[];unknown=[]
 originals=list(lines(page))
 font_aliases={f[3].split('+')[-1]:f[4] for f in page.get_fonts()}
 legend_terms={'base','train_mean','target_mean','offset_mean','affine','eb','target_ridge'}
 legend_right=max((l['bbox'][2] for l,t in lines(page) if t in legend_terms),default=0)
 for line,text in list(lines(page)):
  if text in D:
   translated=D[text]
   rect=fitz.Rect(line['bbox'])
   size=max(s['size'] for s in line['spans'])
   pending.append((rect,translated,size,line['dir'],text))
   page.add_redact_annot(rect,fill=False,cross_out=False)
  elif re.search('[A-Za-z]',text) and not re.fullmatch(r'(?:n|k)=\d+|(?:OCM|STY)-F[01]',text):
   unknown.append(text)
 protected=[]
 for numeric_line,numeric_text in originals:
  if numeric_text not in D:
   nr=fitz.Rect(numeric_line['bbox'])
   if any(nr.intersects(r) for r,_,_,_,_ in pending):
    protected.append(numeric_line)
    page.add_redact_annot(nr,fill=False,cross_out=False)
 page.apply_redactions(images=0,graphics=0)
 page.insert_font(fontname='RestoredNumbers',fontfile=findfont(FontProperties(family='DejaVu Sans')))
 for numeric_line in protected:
  for span in numeric_line['spans']:
   page.insert_text(span['origin'],span['text'],fontname='RestoredNumbers',fontsize=span['size'],color=(0,0,0))
 page.insert_font(fontname='ChineseLabels',fontfile=str(fontpath))
 for rect,target,size,direction,old in pending:
  if not target:continue
  angle=90 if direction[1]<-.5 else 270 if direction[1]>.5 else 0
  if angle:
   area=fitz.Rect(rect.x0-1,rect.y0-1,rect.x1+1,rect.y1+1)
   length=area.height
  else:
   right=legend_right if source.name in ['fig3_learning_curves.pdf','fig6_allocation.pdf'] and old in legend_terms else rect.x1
   area=fitz.Rect(rect.x0-.3,rect.y0-1,right+.3,rect.y1+1)
   length=area.width
  # Retain label centres and fit within the original label footprint.
  target_size=min(size*.9, length/max(font.text_length(target,fontsize=1),1)*.98)
  target_size=max(3.4,target_size)
  result=page.insert_textbox(area,target,fontname='ChineseLabels',fontsize=target_size,
                             lineheight=1,align=fitz.TEXT_ALIGN_LEFT if source.name in ['fig3_learning_curves.pdf','fig6_allocation.pdf'] and old in legend_terms else fitz.TEXT_ALIGN_CENTER,rotate=angle,color=(.08,.08,.08))
  if result<0:
   # Baseline placement avoids text-box font ascent losses in short labels.
   fs=target_size
   if angle==90:
    page.insert_text(fitz.Point(area.x0+fs*.9,area.y1),target,fontname='ChineseLabels',fontsize=fs,rotate=90)
   elif angle==270:
    page.insert_text(fitz.Point(area.x1-fs*.9,area.y0),target,fontname='ChineseLabels',fontsize=fs,rotate=270)
   else:
    width=font.text_length(target,fontsize=fs)
    page.insert_text(fitz.Point(area.x0+(area.width-width)/2,area.y0+fs*.95),target,fontname='ChineseLabels',fontsize=fs)
 new_numbers=nums(page)
 missing=old_numbers-new_numbers
 if missing:
  doc.save(OUT/('debug_'+source.name))
  raise ValueError((source.name,'numeric label removed',missing))
 new_paths=Counter(drawing_signature(d) for d in page.get_drawings())
 missing_paths=old_paths-new_paths
 if missing_paths:raise ValueError((source.name,'plot geometry changed',len(missing_paths)))
 doc.subset_fonts()
 doc.save(OUT/source.name,garbage=4,deflate=True)
 reports.append({'figure':source.name,'translated_labels':len(pending),
                 'numeric_plot_labels_retained':True,'original_vector_paths_retained':True,'remaining_labels':unknown})
 doc.close()

# The design diagram contains no measured data; recreate the same four stages in Chinese.
fig,ax=plt.subplots(figsize=(11.2,3.0))
ax.set_xlim(0,11.2);ax.set_ylim(0,3);ax.axis('off')
texts=[
'文献实验记录\n\n来源按出版条目识别',
'按防泄漏组划分\n外层五折\n\n相关记录不跨训练与测试',
'每个留出来源\n固定测试集 T\n支持候选池 U\n（候选池至少10条）',
'逐步提供 k 条结果\n组成支持集 A_k\n\nk = 0,1,2,3,5,10\n主要比较 k = 5']
colors=['#eaf1f6','#e9f1eb','#fbf1df','#f5e9eb']
for i,(t,c) in enumerate(zip(texts,colors)):
 x=.1+i*2.8
 ax.add_patch(FancyBboxPatch((x,.65),2.52,2.1,boxstyle='round,pad=0.025,rounding_size=0.06',facecolor=c,edgecolor='#71828d',linewidth=.9))
 ax.text(x+1.26,1.7,t,fontproperties=fp,ha='center',va='center',fontsize=11,linespacing=1.55)
 if i<3:ax.add_patch(FancyArrowPatch((x+2.58,1.7),(x+2.77,1.7),arrowstyle='-|>',mutation_scale=11,color='#516777'))
ax.text(5.6,.23,'相同上游测量或相同引用的记录归入同组；标签数量以去重文献记录计量',
        fontproperties=fp,ha='center',va='center',fontsize=10,color='#40515d')
fig.savefig(OUT/'fig1_design.pdf',bbox_inches='tight',pad_inches=.05)
plt.close(fig)
reports.append({'figure':'fig1_design.pdf','recreated_chinese_flow_diagram':True,'measured_data':False})
# Give the dense correspondence diagnostic one readable label per row.
alignment=OUT/'fig5_permutation.pdf'
old=fitz.open(alignment);p=old[0]
centres=sorted(set(round((d['rect'].y0+d['rect'].y1)/2,5) for d in p.get_drawings()
 if d.get('fill') and max(abs(a-b) for a,b in zip(d['fill'],(.1215686,.4666667,.7058824)))<.01
 and 3<d['rect'].height<16))
assert len(centres)==12,centres
new=fitz.open();page=new.new_page(width=p.rect.width+40,height=p.rect.height)
page.show_pdf_page(fitz.Rect(112,0,p.rect.width+40,p.rect.height),old,0,
                   clip=fitz.Rect(72,0,p.rect.width,p.rect.height),keep_proportion=False)
page.insert_font(fontname='AlignmentRowLabels',fontfile=str(fontpath))
labels=[f'{group}：{method}，k={k}' for group in ['OCM F0','OCM F1','STY']
        for k in [5,10] for method in ['仿射','局部岭']]
for y,label in zip(centres,labels):
 size=min(8.5,100/max(font.text_length(label,fontsize=1),1))
 page.insert_textbox(fitz.Rect(2,y-6,106,y+7),label,fontname='AlignmentRowLabels',fontsize=size,
                      lineheight=1,align=fitz.TEXT_ALIGN_RIGHT)
assert not(nums(p)-nums(page))
assert '\x00' not in page.get_text()
new.subset_fonts();assert '\x00' not in page.get_text()
tmp=alignment.with_name('alignment_labels_tmp.pdf')
new.save(tmp,garbage=4,deflate=True);new.close();old.close();tmp.replace(alignment)
for row in reports:
 if row['figure']=='fig5_permutation.pdf':row['row_labels_relaid_for_readability']=True
(ROOT/'中文图表修订_保真检查.json').write_text(json.dumps(reports,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(reports,ensure_ascii=False,indent=2))
