"""Create measured tables, scientific figures and the updated paper sources.

This file performs no fitting or acquisition. Every value comes from audited
historical results or the completed, frozen new-publication application.
"""
from pathlib import Path
import sys
import json
import shutil
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'code'))
from aggregate_results import clustered_interval
from server_guard import write_json, utc

HISTORY = ROOT/'runs/history_v1'
EXTERNAL = ROOT/'runs/new_cohorts/nishimura_measured_v1'
LABELS = {
    'random':'Random', 'operating_vopt':'Operating V',
    'composition_vopt':'Composition V', 'feature_cover':'Feature cover',
    'pool_vopt':'Pool V', 'kernel_vopt':'Kernel V',
    'robust_view_vopt':'Robust multi-view V', 'fixed_best_view':'Fixed historical best',
    'plain_view_gate':'Gate without margin', 'view_gate':'Conservative view gate',
    'shuffled_composition_vopt':'Shuffled composition V'}
POLICIES = tuple(LABELS)
COLORS = {'random':'#7f8c8d','operating_vopt':'#d9853b','composition_vopt':'#2479a7',
          'kernel_vopt':'#7d66b1','robust_view_vopt':'#42958a','view_gate':'#bd4b64'}


def get(frame, task, policy, k=5, method='adaptive_blend', scenario='random'):
    selected=frame[(frame.task==task)&(frame.policy==policy)&(frame.k==k)&(frame.scenario==scenario)]
    assert len(selected)==1,(task,policy,k,scenario)
    return float(selected['mae__'+method].iloc[0])


def savefig(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=180,bbox_inches='tight')
    plt.close(fig)


def contrast(frame,task,left,right,method='adaptive_blend'):
    selected=frame[(frame.task==task)&(frame.scenario=='random')&(frame.k==5)]
    a=selected[selected.policy==left];b=selected[selected.policy==right]
    p=a.merge(b,on=['source','leakage_group'],suffixes=('_left','_right'),validate='one_to_one')
    p['delta']=p['mae__'+method+'_left']-p['mae__'+method+'_right']
    lo,hi=clustered_interval(p,0.95)
    return dict(task=task,left=left,right=right,method=method,
        delta_mae=float(p.delta.mean()),ci95_low=lo,ci95_high=hi,n_sources=len(p),
        analysis_status='post-result exploratory conditional contrast; not an additional confirmatory family')


def table(frame,caption,label,filename):
    lines=[r'\begin{table}[htbp]\centering\small',
        r'\caption{'+caption+r'}\label{'+label+r'}',
        r'\begin{tabular}{lrrrr}\toprule',
        r'Acquisition & Hist. F0 & Hist. F1 & New F0 & New F1 \\ \midrule']
    for row in frame.itertuples():
        lines.append(f'{row.policy_label} & {row.history_f0:.3f} & {row.history_f1:.3f} & {row.new_f0:.3f} & {row.new_f1:.3f} '+r'\\')
    lines += [r'\bottomrule\end{tabular}',
        r'\par\smallskip\footnotesize MAE in percentage points of C$_2$ yield. Frozen adaptive blend, five labels, random query. Hist.: 22 sources, seed-averaged; new: one publication/family, without a population confidence interval.',
        r'\end{table}']
    (ROOT/'manuscripts/tables'/filename).write_text('\n'.join(lines)+'\n')


def main():
    assert json.loads((HISTORY/'verification/audit.json').read_text())['passed']
    assert json.loads((EXTERNAL/'cohort_completion.json').read_text())['passed']
    hist=pd.read_csv(HISTORY/'aggregated/summary.csv')
    ext=pd.read_csv(EXTERNAL/'aggregated/summary.csv')
    source=pd.read_csv(HISTORY/'aggregated/per_source.csv')
    primary=pd.read_csv(HISTORY/'aggregated/primary_family.csv')
    rows=[]
    for policy in POLICIES:
        rows.append(dict(policy=policy,policy_label=LABELS[policy],
            history_f0=get(hist,'ocm_f0',policy),history_f1=get(hist,'ocm_f1',policy),
            new_f0=get(ext,'ocm_f0',policy),new_f1=get(ext,'ocm_f1',policy)))
    matched=pd.DataFrame(rows)
    matched.to_csv(ROOT/'reports/step9_matched_acquisition.csv',index=False)
    contrasts=[]
    for task in ('ocm_f0','ocm_f1'):
        for left,right in [('composition_vopt','operating_vopt'),('composition_vopt','random'),
                           ('composition_vopt','shuffled_composition_vopt'),
                           ('robust_view_vopt','composition_vopt')]:
            contrasts.append(contrast(source,task,left,right))
    pd.DataFrame(contrasts).to_csv(ROOT/'reports/step9_exploratory_mechanism_contrasts.csv',index=False)
    for root,name in [(HISTORY,'history'),(EXTERNAL,'new_publication')]:
        current=pd.read_csv(root/'aggregated/summary.csv')
        values=current[current.scenario.eq('random')&current.k.eq(5)]
        values.to_csv(ROOT/'reports'/f'step9_{name}_all_predictors_k5.csv',index=False)
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                        'pdf.fonttype':42,'font.family':'DejaVu Sans'})
    fig,axes=plt.subplots(1,2,figsize=(10,3.7),sharey=False)
    for ax,data,title in zip(axes,[hist,ext],['Historical development (22 sources)','New published application (one family)']):
        positions=np.arange(2);width=.32
        for offset,policy in [(-.5,'operating_vopt'),(.5,'composition_vopt')]:
            vals=[get(data,t,policy) for t in ('ocm_f0','ocm_f1')]
            bars=ax.bar(positions+offset*width,vals,width,color=COLORS[policy],label=LABELS[policy])
            ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=3,fontsize=9)
        ax.set_xticks(positions,['F0: operating predictor','F1: composition predictor'])
        ax.set_ylabel('MAE (C2-yield percentage points)');ax.set_title(title,fontsize=11)
        ax.set_ylim(0,max(get(data,t,p) for t in ('ocm_f0','ocm_f1') for p in ('operating_vopt','composition_vopt'))*1.23)
    axes[0].legend(frameon=False,fontsize=9,loc='upper right')
    fig.tight_layout();savefig(fig,'step9_01_matched_geometry')
    fig,axes=plt.subplots(2,2,figsize=(10,7.1),sharex=True)
    for ax,(data,task,title) in zip(axes.flat,[(hist,'ocm_f0','Historical F0'),(hist,'ocm_f1','Historical F1'),
                                           (ext,'ocm_f0','New-publication F0'),(ext,'ocm_f1','New-publication F1')]):
        for policy,color in COLORS.items():
            ks=[1,2,3,5,10];vals=[get(data,task,policy,k) for k in ks]
            ax.plot(ks,vals,color=color,label=LABELS[policy],marker='o' if policy=='view_gate' else '.',
                    markersize=4,lw=1.6,ls='--' if policy=='view_gate' else '-')
        ax.set_title(title);ax.set_xticks([1,2,3,5,10]);ax.set_ylabel('MAE (percentage points)')
    for ax in axes[-1]:ax.set_xlabel('Target-source labels')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,-.015),fontsize=9)
    fig.tight_layout(rect=(0,.08,1,1));savefig(fig,'step9_02_budget_curves')
    fig,ax=plt.subplots(figsize=(8.4,3.7))
    for j,row in enumerate(primary.itertuples()):
        ax.plot([row.family_adjusted_ci_low,row.family_adjusted_ci_high],[j,j],color='#2479a7',lw=2)
        ax.plot(row.delta_mae,j,'o',color='#bd4b64',ms=5)
    ax.axvline(0,color='#777777',ls='--',lw=1)
    ax.set_yticks(range(len(primary)),[LABELS[x] for x in primary.control]);ax.invert_yaxis()
    ax.set_xlabel('Conservative gate minus fixed control MAE (percentage points)')
    ax.set_title('Locked historical primary: five labels, F0, random query\n99% conditional intervals; five-comparison adjustment',fontsize=11)
    fig.tight_layout();savefig(fig,'step9_03_primary_family')
    methods=['base','target_mean','local_ridge','adaptive_blend','local_rbf_retuned']
    names=['Unadapted historical model','Support-response mean','Local linear response','Frozen adaptive blend','History-tuned local RBF']
    fig,ax=plt.subplots(figsize=(8.5,3.6))
    vals=[get(ext,'ocm_f1','composition_vopt',method=m) for m in methods]
    bars=ax.barh(range(len(vals)),vals,color=['#9fa9b2','#a9b8c0','#42958a','#2479a7','#7d66b1'])
    ax.set_yticks(range(len(vals)),names);ax.invert_yaxis();ax.bar_label(bars,labels=[f'{v:.3f}' for v in vals],padding=4)
    ax.set_xlim(0,max(vals)*1.15);ax.set_xlabel('New-publication F1 MAE (percentage points)')
    ax.set_title('Label correspondence supports a local response fit\nComposition V; five labels except the unadapted model',fontsize=11)
    fig.tight_layout();savefig(fig,'step9_04_external_predictors')
    table(matched[matched.policy.isin(['random','operating_vopt','composition_vopt','kernel_vopt','robust_view_vopt','fixed_best_view','view_gate'])],
          'Matched acquisition and prediction representations in the executed OCM follow-up.',
          'tab:step9matched','step9_matched.tex')
    table(matched,'All eleven acquisition policies in the executed OCM follow-up.',
          'tab:step9all','step9_all_policies.tex')
    p=[r'\begin{table}[htbp]\centering\small',r'\caption{Locked historical primary contrasts for the conservative view gate.}\label{tab:step9primary}',
       r'\begin{tabular}{lrrr}\toprule Fixed control & $\Delta$ MAE & 99\% lower & 99\% upper \\ \midrule']
    for r in primary.itertuples():p.append(f'{LABELS[r.control]} & {r.delta_mae:.4f} & {r.family_adjusted_ci_low:.4f} & {r.family_adjusted_ci_high:.4f} '+r'\\')
    p += [r'\bottomrule\end{tabular}\par\smallskip\footnotesize Candidate minus control; negative favors the gate. Twenty-two equally weighted sources. No contrast supports gate superiority after the five-comparison adjustment. Fixed historical best and composition V coincide here.\end{table}']
    (ROOT/'manuscripts/tables/step9_primary.tex').write_text('\n'.join(p)+'\n')
    facts=dict(created_utc=utc(),historical_parts=25,historical_sources=22,
        historical_cells=93150,historical_scalar_predictions=7948800,
        new_backbone_fits=56,new_outer_backbone_fits_in_historical_comparison=0,
        external_measured_records=1739,external_catalysts=116,external_sources=1,
        external_cells=1380,external_scalar_predictions=7443720,
        gate_primary_mae=float(primary.candidate_mae.iloc[0]),
        composition_primary_mae=float(primary.loc[primary.control.eq('composition_vopt'),'control_mae'].iloc[0]),
        primary_advantage_supported=bool(primary.adjusted_support_for_advantage.any()),
        historical_primary_gate_composition_choices=109,historical_primary_gate_episodes=110,
        external_f1_composition_blend_mae=get(ext,'ocm_f1','composition_vopt'),
        external_f1_random_blend_mae=get(ext,'ocm_f1','random'),
        external_f1_composition_vs_random_reduction_percent=100*(1-get(ext,'ocm_f1','composition_vopt')/get(ext,'ocm_f1','random')),
        external_f0_operating_mae=get(ext,'ocm_f0','operating_vopt'),external_f0_composition_mae=get(ext,'ocm_f0','composition_vopt'),
        external_f1_operating_mae=get(ext,'ocm_f1','operating_vopt'),
        external_f1_local_rbf_composition_mae=get(ext,'ocm_f1','composition_vopt',method='local_rbf_retuned'),
        scope='historical development plus one frozen published-measurement application; no new wet-lab experiment or population CI from one family')
    write_json(ROOT/'reports/step9_measured_facts.json',facts)
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()
