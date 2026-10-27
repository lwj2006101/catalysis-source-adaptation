"""Final evidence tables/figures: no scientific prediction is rerun here."""
from pathlib import Path
import os,json,shutil
os.environ.setdefault('MPLCONFIGDIR',str(Path(__file__).resolve().parents[1]/'tmp/mpl'))
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[1];REF=ROOT/'reference_step10'
P7=REF/'reference_step9/reference_step7'
LABELS={'random':'Random','composition_vopt':'Composition V','operating_vopt':'Operating V',
        'pool_vopt':'Pool V','expanded_centered':'Expanded-kernel risk',
        'rbf_matched':'RBF-matched risk','rbf_amplification_robust':'Penalized RBF risk'}


def dump(n,v):
    p=ROOT/'reports'/n;p.parent.mkdir(parents=True,exist_ok=True)
    p.write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def table(name,headings,rows,align):
    s=['\\begin{tabular}{'+align+'}\\toprule',' & '.join(headings)+r' \\ \midrule']
    s += [' & '.join(str(x) for x in row)+r' \\' for row in rows]
    s += ['\\bottomrule\\end{tabular}\n']
    (ROOT/'manuscripts/tables'/name).write_text('\n'.join(s))


def save(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight',dpi=220);plt.close(fig)


def main():
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
        'axes.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42})
    for name in ['fig1_split_audit','step7_01_primary','step9_01_matched_geometry','step9_04_external_predictors']:
        for extension in ['pdf','png']:
            source=REF/'figures'/f'{name}.{extension}'
            if source.exists():shutil.copy2(source,ROOT/'figures'/source.name)
    original=pd.read_csv(P7/'results/primary_family.csv')
    for name in ['internal_k5_body.tex','external_k5_body.tex','step7_primary.tex']:
        shutil.copy2(REF/'manuscripts/tables'/name,ROOT/'manuscripts/tables'/name)
    alignment=pd.read_csv(P7/'results/label_permutation_summary.csv')
    alignment=alignment[(alignment.scenario=='random')&(alignment.policy=='linear_vopt')]
    rows=[]
    panel_names={'methanol_conversion':'Conversion','methanol_selectivity':'Selectivity',
                 'methanol_sty':'STY','ocm_f0':'OCM F0','ocm_f1':'OCM F1'}
    for f in alignment.itertuples():
        rows.append([panel_names[f.endpoint],f'{f.aligned_mae:.3f}',f'{f.permuted_mae:.3f}',
                     f'{f.delta_mae:.3f}',f'[{f.ci_low:.3f}, {f.ci_high:.3f}]'])
    table('label_alignment.tex',['Panel','Aligned','Permuted','$\\Delta$ MAE','95\\% interval'],rows,'lrrrl')
    fig,ax=plt.subplots(figsize=(7.8,3.2));y=np.arange(len(alignment));d=alignment.delta_mae.to_numpy()/alignment.aligned_mae.to_numpy()*100
    low=alignment.ci_low.to_numpy()/alignment.aligned_mae.to_numpy()*100;high=alignment.ci_high.to_numpy()/alignment.aligned_mae.to_numpy()*100
    ax.errorbar(d,y,xerr=[d-low,high-d],fmt='o',color='#27677a',capsize=4)
    ax.set_yticks(y,[panel_names[x] for x in alignment.endpoint]);ax.invert_yaxis();ax.axvline(0,color='#777777',lw=1)
    ax.set_xlabel('Permutation MAE increase / observed aligned MAE (%)');ax.spines[['top','right']].set_visible(False)
    ax.grid(axis='x',alpha=.18);fig.tight_layout();save(fig,'label_correspondence')
    ha=json.loads((ROOT/'runs/operator_history_v1/verification/audit.json').read_text())
    oa=json.loads((ROOT/'runs/operator_opened_v1/verification/audit.json').read_text())
    assert ha['passed'] and oa['passed']
    hist=pd.read_csv(ROOT/'runs/operator_history_v1/aggregated/summary.csv')
    part=pd.read_csv(ROOT/'runs/operator_opened_v1/aggregated/per_partition.csv')
    source=pd.read_csv(ROOT/'runs/operator_history_v1/aggregated/per_source.csv')
    part=part[part.k.eq(5)]
    reference=part[part.policy.eq('random')][['partition','mae__fixed_rbf']].rename(columns={'mae__fixed_rbf':'random_mae'})
    pair=part.merge(reference,on='partition',validate='many_to_one');pair['delta_mae']=pair.mae__fixed_rbf-pair.random_mae
    pair['relative_gain_percent']=100*(1-pair.mae__fixed_rbf/pair.random_mae)
    pair.to_csv(ROOT/'reports/repeated_partition_contrasts.csv',index=False)
    policies=list(LABELS);rows=[];stats={}
    for policy in policies:
        f=pair[pair.policy==policy]
        stat=dict(mean_mae=float(f.mae__fixed_rbf.mean()),mean_delta=float(f.delta_mae.mean()),
                  median_delta=float(f.delta_mae.median()),minimum_delta=float(f.delta_mae.min()),
                  maximum_delta=float(f.delta_mae.max()),mean_relative_gain_percent=float(f.relative_gain_percent.mean()),
                  improved_partitions=int((f.delta_mae < -1e-10).sum()),n_partitions=len(f))
        stats[policy]=stat
        rows.append([LABELS[policy],f'{stat["mean_mae"]:.3f}',f'{stat["mean_delta"]:+.3f}',
                     f'{stat["minimum_delta"]:+.3f} to {stat["maximum_delta"]:+.3f}',
                     str(stat['improved_partitions'])+'/20'])
    table('repeated_catalyst.tex',['Acquisition','Mean MAE','Mean $\\Delta$','Partition $\\Delta$ range','Improved'],rows,'lrrll')
    fig,axes=plt.subplots(1,2,figsize=(8.4,3.5),gridspec_kw={'width_ratios':[1.25,1]})
    drawpol=['composition_vopt','pool_vopt','expanded_centered','rbf_matched']
    ys=[pair[pair.policy==p].delta_mae.to_numpy() for p in drawpol]
    axes[0].boxplot(ys,vert=False,tick_labels=[LABELS[p] for p in drawpol],showfliers=False,
                    boxprops={'color':'#27677a'},medianprops={'color':'#d7833f'})
    for j,vals in enumerate(ys):
        offsets=np.linspace(-.16,.16,len(vals));axes[0].scatter(vals,j+1+offsets,s=13,color='#27677a',alpha=.65)
    axes[0].axvline(0,color='#777777',lw=1);axes[0].set_xlabel('MAE minus random (pp)')
    axes[0].set_title('A  Twenty catalyst-disjoint partitions',loc='left')
    opened=pd.read_csv(REF/'runs/applications/nishimura_opened_v1/aggregated/summary.csv')
    opened=opened[(opened.task=='ocm_f1')&(opened.k==5)]
    plotpol=['random','composition_vopt','centered_mse'];x=np.arange(3)
    for j,scenario in enumerate(['random','catalyst_holdout']):
        f=opened[opened.scenario==scenario].set_index('policy')
        axes[1].bar(x+(j-.5)*.34,f.loc[plotpol,'mae__locked_local_rbf'],width=.34,
                    color=['#27677a','#d7833f'][j],label=['Ordinary query','Initial catalyst holdout'][j])
    axes[1].set_xticks(x,['Random','Comp. V','Expanded\nrisk']);axes[1].set_ylabel('MAE (pp)')
    axes[1].set_ylim(0,6.4)
    axes[1].set_title('B  Query definition changes rankings',loc='left');axes[1].legend(frameon=False,fontsize=8)
    for ax in axes:ax.spines[['top','right']].set_visible(False);ax.grid(axis='x' if ax==axes[0] else 'y',alpha=.14);ax.set_axisbelow(True)
    fig.tight_layout();save(fig,'task_sensitivity')
    # Rank diagnostics within each source: nuisance response scales are not pooled.
    ss=source[(source.scenario=='random')&(source.k==5)].copy()
    means=[]
    for src,f in ss.groupby('source'):
        rr=f.nominal_risk.rank(method='average');er=f.mse__fixed_rbf.rank(method='average')
        means.append(dict(source=src,spearman=float(rr.corr(er))))
    pd.DataFrame(means).to_csv(ROOT/'reports/working_risk_rank_correlations.csv',index=False)
    eps=[json.loads(p.read_text())['epsilon'] for p in (ROOT/'runs/operator_history_v1/parts').glob('*/parameters.json')]
    matched=hist[(hist.scenario=='random')&(hist.k==5)].set_index('policy')
    same=source[(source.scenario=='random')&(source.k==5)&source.policy.isin(['rbf_matched','rbf_amplification_robust'])]
    delta=same.pivot(index=['source','leakage_group'],columns='policy',values='mae__fixed_rbf')
    delta['delta']=delta.rbf_amplification_robust-delta.rbf_matched
    draw=np.random.default_rng(111103).integers(0,len(delta),(20000,len(delta)))
    boot=delta.delta.to_numpy()[draw].mean(1);interval=np.quantile(boot,[.025,.975])
    delta.to_csv(ROOT/'reports/robust_vs_nominal_per_source.csv')
    rows=[]
    for policy in ['random','composition_vopt','centered_mse','rbf_matched','rbf_amplification_robust']:
        f=matched.loc[policy]
        name=LABELS.get(policy,'Expanded-kernel risk')
        rows.append([name,f'{f.mae__fixed_rbf:.3f}',f'{f.nominal_risk:.3f}',f'{f.amplification:.3f}'])
    table('matched_rbf.tex',['Acquisition','MAE (pp)','Working risk','Amplification'],rows,'lrrr')
    facts=dict(historical_audit=ha,opened_audit=oa,actual_new_scalar_predictions=ha['scalar_predictions']+oa['scalar_predictions'],
       new_backbone_fits=0,additional_kernel_acquisition_tuning_actually_executed=True,
       risk_penalty_selection_counts={str(x):eps.count(x) for x in sorted(set(eps))},
       full_history_epsilon=json.loads((ROOT/'runs/operator_full_history_v1/parameters.json').read_text())['epsilon'],
       fixed_rbf_history_k5_mae={p:float(matched.loc[p,'mae__fixed_rbf']) for p in matched.index},
       robust_minus_nominal_historical_delta=float(delta.delta.mean()),
       robust_minus_nominal_historical_nominal95_interval=interval.tolist(),
       repeated_catalyst=stats,n_sources_repeated_study=1,n_experiment_families_repeated_study=1,
       new_independent_measured_families=0,working_risk_rank_spearman_median=float(np.median([m['spearman'] for m in means])))
    dump('step11_measured_facts.json',facts)
    macros=[]
    for key,value in [('NewScalarCount',facts['actual_new_scalar_predictions']),('HistoricalCells',ha['cells']),
                      ('OpenedCells',oa['cells']),('RbfMatchMae',matched.loc['rbf_matched','mae__fixed_rbf']),
                      ('RobustMae',matched.loc['rbf_amplification_robust','mae__fixed_rbf'])]:
        macros.append('\\newcommand{\\'+key+'}{'+(f'{value:,.0f}' if 'Count' in key or 'Cells' in key else f'{value:.3f}')+'}')
    (ROOT/'manuscripts/result_macros.tex').write_text('\n'.join(macros)+'\n')
    print(json.dumps(facts,indent=2))


if __name__=='__main__':main()
