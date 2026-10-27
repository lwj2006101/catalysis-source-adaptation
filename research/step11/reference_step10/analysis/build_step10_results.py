"""Build tables, scientific figures and manuscripts from audited actual runs."""
from pathlib import Path
import json
import math
import os
os.environ.setdefault('MPLCONFIGDIR', '/workspace/scratch/8bea8356e6d0/tmp/step10_mpl')
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    'random':'Random', 'operating_vopt':'Operating V', 'composition_vopt':'Composition V',
    'feature_cover':'Feature cover', 'pool_vopt':'Pool V', 'kernel_vopt':'Earlier kernel V',
    'robust_view_vopt':'Robust multi-view V', 'finite_gp_v':'Finite-intercept GP',
    'centered_mse':'Centered squared risk', 'centered_mae':'Centered absolute risk',
    'hedged_mse':'Averaged squared risk', 'hedged_mae':'Averaged absolute risk'}
ORDER = list(LABELS)
METHODS = ['matched_kernel','locked_local_rbf','frozen_blend','local_ridge',
           'target_mean','zero_label','random_tuned_isotropic']


def load(path):
    return json.loads((ROOT / path).read_text())


def write_json(path, value):
    (ROOT / path).parent.mkdir(parents=True, exist_ok=True)
    (ROOT / path).write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n')


def table(path, caption, label, headings, rows, align=None):
    align = align or 'l' + 'r'*(len(headings)-1)
    text = ['\\begin{table}[htbp]\\centering\\small', '\\caption{'+caption+'}\\label{'+label+'}',
            '\\begin{tabular}{'+align+'}\\toprule', ' & '.join(headings)+r' \\ \midrule']
    text += [' & '.join(map(str,row))+r' \\' for row in rows]
    text += ['\\bottomrule\\end{tabular}','\\end{table}\n']
    (ROOT / 'manuscripts/tables' / path).write_text('\n'.join(text))


def ci(paired):
    totals = paired.groupby('leakage_group').delta.agg(['sum','count'])
    draw = np.random.default_rng(31010).integers(0,len(totals),(20000,len(totals)))
    boot = totals['sum'].to_numpy()[draw].sum(1)/totals['count'].to_numpy()[draw].sum(1)
    return [float(v) for v in np.quantile(boot,[.025,.975])]


def secondary(per_source):
    f=per_source[(per_source.task=='ocm_f1') & (per_source.scenario=='random') & (per_source.k==5)]
    contrasts=[('Centered squared vs random','centered_mse','matched_kernel','random','matched_kernel'),
               ('Centered squared vs finite GP','centered_mse','matched_kernel','finite_gp_v','matched_kernel'),
               ('Centered absolute vs squared','centered_mae','matched_kernel','centered_mse','matched_kernel'),
               ('Averaged squared vs centered','hedged_mse','matched_kernel','centered_mse','matched_kernel'),
               ('Expanded vs isotropic (composition V)','composition_vopt','matched_kernel','composition_vopt','random_tuned_isotropic'),
               ('Expanded vs strong RBF (composition V)','composition_vopt','matched_kernel','composition_vopt','locked_local_rbf')]
    rows=[]
    for name,p1,m1,p0,m0 in contrasts:
        a=f[f.policy==p1][['source','leakage_group','mae__'+m1]].rename(columns={'mae__'+m1:'candidate'})
        b=f[f.policy==p0][['source','leakage_group','mae__'+m0]].rename(columns={'mae__'+m0:'control'})
        pair=a.merge(b,on=['source','leakage_group'],validate='one_to_one')
        pair['delta']=pair.candidate-pair.control
        lo,hi=ci(pair)
        rows.append(dict(contrast=name,delta_mae=float(pair.delta.mean()),ci_low=lo,ci_high=hi,
                         n_sources=len(pair),status='post-hoc development; nominal 95%, no family adjustment'))
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/'reports/step10_secondary_contrasts.csv',index=False)
    return result


def catalyst_check():
    p=ROOT/'runs/applications/nishimura_opened_v1/parts/cohort'
    data=pd.read_csv(p/'evaluated_input.csv.gz')
    ledger=pd.read_csv(p/'support_ledger.csv.gz')
    provenance=pd.read_csv(ROOT/'reference_step9/data/curated/nishimura_measured_v1/row_provenance.csv')
    lookup=provenance.set_index('row_id').catalyst_id
    records={}
    for scenario,f in ledger.groupby('scenario'):
        q=json.loads(f.iloc[0].query_indices); s=json.loads(f.iloc[0].pool_indices)
        qids=data.row_id.iloc[q].tolist(); pids=data.row_id.iloc[s].tolist()
        qcat=set(lookup.loc[qids]); pcat=set(lookup.loc[pids])
        records[scenario]=dict(query_rows=len(q),pool_rows=len(s),query_catalysts=len(qcat),
            pool_catalysts=len(pcat),shared_catalysts=len(qcat&pcat),
            query_catalyst_ids=sorted(qcat),pool_catalyst_ids=sorted(pcat),
            query_row_ids=qids,pool_row_ids=pids)
        assert set(q).isdisjoint(s) and len(q)+len(s)==1739
    assert records['catalyst_holdout']['shared_catalysts']==0
    assert records['catalyst_holdout']['query_catalysts']==29
    assert records['catalyst_holdout']['pool_catalysts']==87
    write_json('reports/step10_catalyst_partition_audit.json',dict(passed=True,
               n_publications=1,n_experiment_families=1,previously_opened=True,scenarios=records))
    return {k:{q:v for q,v in d.items() if q.endswith('rows') or q.endswith('catalysts')}
            for k,d in records.items()}


def savefig(fig, name):
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight')
    fig.savefig(ROOT/'figures'/f'{name}.png',bbox_inches='tight',dpi=220)
    plt.close(fig)


def plots(cases, primary):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.titlesize':10,
                         'axes.labelsize':10,'legend.fontsize':9,'pdf.fonttype':42})
    policies=['random','composition_vopt','pool_vopt','centered_mse','centered_mae','hedged_mae']
    tick=['Random','Comp. V','Pool V','Center\nMSE','Center\nMAE','Avg.\nMAE']
    fig,axes=plt.subplots(2,2,figsize=(8.3,6.5),sharey=False)
    for ax,(title,f) in zip(axes.flat,cases):
        f=f.set_index('policy'); x=np.arange(len(policies))
        ax.bar(x-.18,f.loc[policies,'mae__matched_kernel'],width=.35,color='#27677a',label='Selected expanded kernel')
        ax.bar(x+.18,f.loc[policies,'mae__locked_local_rbf'],width=.35,color='#d78b4a',label='Strong local RBF')
        ax.set_xticks(x,tick,fontsize=9); ax.set_ylabel('MAE (percentage points)')
        ax.set_title(title,loc='left',weight='bold'); ax.set_ylim(0,6.6)
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='y',alpha=.16); ax.set_axisbelow(True)
    handles,labels=axes.flat[0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='upper center',bbox_to_anchor=(.54,1.015),ncol=2,frameon=False)
    fig.tight_layout(pad=1.1,h_pad=1.7,rect=[0,0,1,.965])
    savefig(fig,'step10_01_task_conditioned')

    riskpol=['finite_gp_v','centered_mse','centered_mae','hedged_mse','hedged_mae']
    risknames=['Finite GP','Centered MSE','Centered MAE','Averaged MSE','Averaged MAE']
    fig,axes=plt.subplots(1,3,figsize=(8.3,3.3),sharey=True)
    for ax,(title,f) in zip(axes,[cases[0],cases[1],cases[2]]):
        f=f.set_index('policy'); y=np.arange(5)
        ax.plot(f.loc[riskpol,'mae__matched_kernel'],y,'o-',color='#27677a',lw=1.2)
        ax.set_yticks(y,risknames); ax.invert_yaxis(); ax.set_xlim(2.2,6.2)
        ax.set_xlabel('MAE (pp)'); ax.set_title(title.split('\n')[0],fontsize=9,loc='left')
        ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.2)
    fig.tight_layout(pad=1.1)
    savefig(fig,'step10_02_risk_ablation')

    fig,ax=plt.subplots(figsize=(8.1,3.6))
    y=np.arange(len(primary)); d=primary.delta_mae.to_numpy()
    ax.errorbar(d,y,xerr=np.array([d-primary.family_adjusted_ci_low.to_numpy(),
                     primary.family_adjusted_ci_high.to_numpy()-d]),fmt='o',color='#27677a',capsize=4)
    ax.set_yticks(y,[LABELS[x] for x in primary.control]);ax.invert_yaxis()
    ax.axvline(0,color='#808080',lw=1);ax.set_xlabel('Averaged absolute-risk MAE minus control (pp)')
    ax.set_title('Historical F1, k = 5: conditional 99% intervals',loc='left',weight='bold')
    ax.spines[['top','right']].set_visible(False);ax.grid(axis='x',alpha=.18)
    fig.tight_layout()
    savefig(fig,'step10_03_primary_family')

    fig,axes=plt.subplots(1,3,figsize=(8.3,3.5),sharey=False)
    colors=['#666666','#27677a','#68a79a','#de9a4a','#a66a83','#8c80b3']
    for ax,(title,f) in zip(axes,[cases[0],cases[1],cases[2]]):
        for policy,col in zip(policies,colors):
            d=f.attrs['full']; d=d[d.policy==policy].sort_values('k')
            ax.plot(d.k,d.mae__matched_kernel,'o-',color=col,lw=1.5,ms=3,label=LABELS[policy])
        ax.set_xticks([1,2,3,5,10]);ax.set_xlabel('Support labels');ax.set_ylabel('MAE (pp)')
        ax.set_title(title.split('\n')[0],fontsize=9,loc='left'); ax.spines[['top','right']].set_visible(False)
        ax.grid(alpha=.15)
    h,l=axes[0].get_legend_handles_labels();fig.legend(h,l,loc='lower center',ncol=3,frameon=False,
        bbox_to_anchor=(.5,-.075),fontsize=8.5)
    fig.tight_layout(rect=[0,.09,1,1],pad=1)
    savefig(fig,'step10_04_budget_curves')


def main():
    hroot=ROOT/'runs/history_v1'; eroot=ROOT/'runs/applications/nishimura_opened_v1'
    audits=[load(str(p.relative_to(ROOT))+'/verification/audit.json') for p in [hroot,eroot]]
    assert all(a['passed'] for a in audits)
    hist=pd.read_csv(hroot/'aggregated/summary.csv'); ext=pd.read_csv(eroot/'aggregated/summary.csv')
    primary=pd.read_csv(hroot/'aggregated/primary_family.csv')
    sec=secondary(pd.read_csv(hroot/'aggregated/per_source.csv'))
    parts=[load(str(p.relative_to(ROOT))) for p in (hroot/'parts').glob('*/completed.json')]
    ec=load('runs/applications/nishimura_opened_v1/application_completion.json')
    partition=catalyst_check()
    choices=[]
    for p in sorted((hroot/'parts').glob('*/parameters.json')):
        record=json.loads(p.read_text()); tasks=record['tasks']
        for task,t in tasks.items():
            w=np.array(t['working_weights']); idx=int(np.argmin(t['historical_group_equal_mae'][:9]))
            choices.append(dict(part=p.parent.name,task=task,theta=json.dumps(t['theta'],sort_keys=True),
                kind=t['theta']['kind'],isotropic_theta=json.dumps(t['atlas'][idx],sort_keys=True),
                working_noise=t['working_noise'],weight_entropy=float(-np.sum(w*np.log(np.maximum(w,1e-300)))),
                n_historical_groups=t['n_historical_groups']))
    choices=pd.DataFrame(choices);choices.to_csv(ROOT/'reports/step10_kernel_choices.csv',index=False)
    fullparams=load('runs/full_history_v1/parameters.json')
    facts=dict(historical_parts=len(parts),historical_cells=sum(p['n_cells'] for p in parts),
        historical_query_records=sum(p['n_prediction_rows'] for p in parts),
        historical_scalar_predictions=sum(p['n_scalar_predictions'] for p in parts),
        application_cells=ec['n_cells'],application_query_records=ec['n_query_records'],
        application_scalar_predictions=ec['scalar_predictions'],new_backbone_refits=0,
        historical_audit=audits[0],application_audit=audits[1],math_checks=load('runs/verification/math.json'),
        primary_family=primary.to_dict('records'),primary_candidate_advantage_supported=False,
        catalyst_partition=partition,application_rows=1739,application_catalysts=116,
        new_independent_publications_this_step=0,new_wet_lab_experiments=0,
        selected_kernel_family_counts=choices[choices.task=='ocm_f1'].kind.value_counts().to_dict(),
        final_F1_theta=fullparams['tasks']['ocm_f1']['theta'],
        final_isotropic_theta=fullparams['tasks']['ocm_f1']['atlas'][int(np.argmin(
            fullparams['tasks']['ocm_f1']['historical_group_equal_mae'][:9]))],
        final_historical_kernel_validation_scalar_values=fullparams['actual_kernel_validation_prediction_values'])
    assert facts['historical_parts']==25 and not primary.adjusted_support_for_advantage.any()
    facts['total_scalar_predictions']=facts['historical_scalar_predictions']+facts['application_scalar_predictions']

    cases=[]
    for title,frame,scenario in [('History (22 sources)\nRandom queries',hist,'random'),
                                  ('Opened source: random\nOne family',ext,'random'),
                                  ('Opened source: catalyst holdout\n29 query catalysts / 87 pool catalysts',ext,'catalyst_holdout'),
                                  ('Opened source: high temperature\nOne family',ext,'temperature_high')]:
        f=frame[(frame.task=='ocm_f1')&(frame.scenario==scenario)&(frame.k==5)].copy()
        f.attrs['full']=frame[(frame.task=='ocm_f1')&(frame.scenario==scenario)].copy()
        cases.append((title,f))
    values={}
    for key,(_,f) in zip(['history_random','opened_random','opened_catalyst','opened_temperature'],cases):
        values[key]={p:{m:float(f.set_index('policy').loc[p,'mae__'+m]) for m in METHODS} for p in ORDER}
    facts['F1_k5_mae']=values
    facts['catalyst_strong_RBF_centered_vs_random_pct']=100*(1-values['opened_catalyst']['centered_mse']['locked_local_rbf']/values['opened_catalyst']['random']['locked_local_rbf'])
    write_json('reports/step10_measured_facts.json',facts)

    rows=[]
    mainpol=['random','composition_vopt','pool_vopt','finite_gp_v','centered_mse','centered_mae','hedged_mae']
    for p in mainpol:
        rows.append([LABELS[p],'Expanded kernel']+[f'{values[c][p]["matched_kernel"]:.3f}' for c in ['history_random','opened_random','opened_catalyst']])
    for p in ['random','composition_vopt','centered_mse']:
        rows.append([LABELS[p],'Strong local RBF']+[f'{values[c][p]["locked_local_rbf"]:.3f}' for c in ['history_random','opened_random','opened_catalyst']])
    table('step10_matched.tex','Five-label F1 comparisons with predictor parameters held fixed across policies. MAE is in percentage points. History is development over 22 sources; both publication applications use the same previously opened family. The catalyst query and pool have disjoint reported catalyst identifiers.',
          'tab:step10matched',['Acquisition','Predictor','History','Opened random','Catalyst holdout'],rows,'llrrr')
    rows=[[LABELS[r.control],f'{r.delta_mae:+.3f}',f'[{r.family_adjusted_ci_low:+.3f}, {r.family_adjusted_ci_high:+.3f}]','No'] for r in primary.itertuples()]
    table('step10_primary.tex',r'Locked development primary family: averaged absolute-risk selection minus control, with the same history-selected expanded F1 kernel, five labels and random queries. Negative favors the candidate. Conditional 99\% intervals adjust the five comparisons; none supports superiority.',
          'tab:step10primary',['Control','$\\Delta$ MAE','99\\% interval','Advantage'],rows,'lrrc')
    text=[]
    for i,((title,f),key) in enumerate(zip(cases,values)):
        rows=[[LABELS[p]]+[f'{values[key][p][m]:.3f}' for m in ['matched_kernel','locked_local_rbf','random_tuned_isotropic']] for p in ORDER]
        name=f'step10_all_{i}.tex'
        table(name,'All twelve acquisitions at five labels, F1: '+title.split('\n')[0]+'. MAE in percentage points. The isotropic control uses the same historical random-support tuning draws as the expanded kernel. All seven predictors and both representations are available in electronic tables.',
              'tab:step10all'+str(i),['Acquisition','Expanded','Strong RBF','Same-tuning isotropic'],rows,'lrrr')
        text.append('\\input{tables/'+name+'}')
    (ROOT/'manuscripts/tables/step10_all_policies.tex').write_text('\n'.join(text)+'\n')
    rows=[[r.contrast,f'{r.delta_mae:+.3f}',f'[{r.ci_low:+.3f}, {r.ci_high:+.3f}]'] for r in sec.itertuples()]
    table('step10_secondary.tex',r'Post-hoc historical F1 contrasts at five labels with random queries. Negative favors the first named method. These nominal 95\% conditional intervals have no multiplicity adjustment and do not replace the locked primary family.',
          'tab:step10secondary',['Contrast','$\\Delta$ MAE','Nominal 95\\% interval'],rows,'lrr')
    counts=facts['selected_kernel_family_counts']
    rows=[[kind,counts.get(kind,0)] for kind in ['isotropic','additive','product','anova']]
    table('step10_kernel_choices.tex','History-only selection frequency for the expanded F1 predictor in the 25 outer-training fits. Choices favor block kernels, but that frequency does not establish superiority on held-out responses. Full-history choice is additive with $\\alpha=0.01$, $\\gamma_O=0.1$, $\\gamma_C=10$.',
          'tab:step10kernels',['Kernel family','Number of outer histories'],rows,'lr')
    plots(cases,primary)

    results=r'''\subsection{Matched working risk helps conditionally, rather than universally}
The second development grid separates additional predictor flexibility from acquisition changes (Table~\ref{tab:step10matched}; Fig.~\ref{fig:step10task}). With the same expanded F1 kernel, five-label historical random-query MAE is 3.317 for random supports, 2.863 for composition V, and 2.792 for centered squared-risk selection. Finite-intercept GP selection gives 2.790, essentially the same result; removing its finite intercept prior therefore does not account for the point improvement. Centered absolute-risk selection gives 2.845 and does not improve on squared-risk selection. The expanded predictor is selected from block kernels in all twenty-five histories, but its improvement over the same-tuning isotropic predictor is small: under composition V, the two errors are 2.863 and 2.897. Kernel-family selection frequency alone is not a performance result.

The locked covariance-averaged absolute-risk candidate has historical MAE 3.030. None of its five conditional family-adjusted comparisons supports superiority; composition V and robust multi-view V give 2.863 and 2.859. Covariance averaging and a loss-matched objective do not supply a validated new-algorithm advantage. The primary family remains intact, while secondary contrasts and all predictor combinations are reported in the SI.

\input{tables/step10_matched.tex}
\plotfigure{step10_01_task_conditioned}{Selection and predictor effects under distinct query tasks at five labels. Within a task, each predictor's settings are fixed across all policies. The first panel is historical development; the other panels use the same opened publication. Better historical working-risk acquisition does not preserve its ranking under every query partition.}{fig:step10task}

The catalyst stress query comprises 435 measurements for 29 catalyst identifiers, with all 1304 support-pool rows belonging to the other 87 identifiers. In contrast, the earlier random partition has all 116 identities represented in both query and pool. Under catalyst holdout, centered squared-risk selection gives expanded-kernel MAE 4.199, compared with 4.915 for random and 5.055 for composition V. On exactly those same support sets, the strong isotropic RBF is better still: 3.689, compared with 4.818 for random and 4.680 for composition V. This is a 23.4\% point improvement over random for an unchanged established predictor. Finite-intercept, centered squared and centered absolute selection choose the same five supports in this partition; the benefit cannot be assigned specifically to free centering or absolute risk.

The regular random query gives the opposite ordering. Expanded-kernel centered squared selection has MAE 5.460, worse than random at 4.848 and composition V at 4.164; the corresponding strong-RBF errors are 5.145, 4.792 and 3.813. The wider kernel family also fails to improve on that RBF in the opened applications. Thus acquisition must be evaluated for the deployed predictor and query task; a lower working variance is not evidence of a universal reduction in actual error. These are single-family stress/development point estimates, without a population confidence interval. Reported-identifier separation does not certify unseen synthesis, physical batches or laboratories.

All 25 new historical comparisons and all three opened-publication partitions were executed. They add 82,350 and 1,830 evaluation cells, respectively, and 9,795,380 total scalar query predictions. Every scalar loss and all 16,836 support orders are audited; separate kernel/sklearn checks reconstruct 825,840 query values. Mathematical tests also independently verify the free-intercept weights and working-risk covariance. No global backbones were newly fitted in this extension: the earlier frozen models are reused, while kernel tuning, designs and predictions are new executed work.
'''
    (ROOT/'manuscripts/step10_results_main.tex').write_text(results)
    update_manuscripts()
    print('Built measured facts, electronic/LaTeX tables, four scientific figures and updated manuscript sources.')


def update_manuscripts():
    original=ROOT/'reference_step9/manuscripts'
    p=ROOT/'manuscripts/manuscript_en.tex'
    text=(original/p.name).read_text()
    start=text.index('\\begin{abstract}')+len('\\begin{abstract}')
    end=text.index('\\end{abstract}',start)
    abstract='''
Target-source labels can recover catalytic prediction accuracy, while their measurement locations determine the value of the same budget. We separate response level, input--label correspondence, historical response reuse, acquisition representation and deployed-predictor risk under publication shift. Source-disjoint analyses cover four outcomes in two reaction systems, represented by five panels. At five labels, established V-optimal acquisition reduces an unchanged blend's source-equal mean absolute error from 62.608 to 50.274 in native methanol STY units and from 3.081 to 2.708 percentage points for composition-enriched oxidative coupling of methane (OCM); both earlier primary gains survive five-comparison adjustment. Representation crossing shows that chemistry can inform acquisition even when a predictor consumes only operating features. A pipeline frozen before its first application to 1739 author-reported measurements for 116 catalysts retains a 12.0\\% composition-acquisition point improvement for the composition blend, while a strong local RBF performs better. A subsequent development study crosses twelve acquisitions with seven fixed predictors, coupling standard free-intercept kernel regression to squared, absolute and covariance-averaged working risks. In catalyst-disjoint stress within that opened publication, matched-kernel acquisition reduces an unchanged RBF's five-label MAE from 4.818 to 3.689, but worsens ordinary random-query prediction. The covariance-averaged primary candidate does not outperform strong controls. These results establish conditional information and equal-budget acquisition benefits, while identifying where predictor, query task and working-risk assumptions constrain them.
'''
    text=text[:start]+abstract+text[end:]
    text=text.replace('\\input{step9_methods_main.tex}','\\input{step9_methods_main.tex}\n\\input{step10_methods_main.tex}')
    text=text.replace('\\input{step9_results_main.tex}','\\input{step9_results_main.tex}\n\\input{step10_results_main.tex}')
    text=text.replace('Their informative and unsuccessful combinations are retained',
        'A predictor-matched extension evaluates source-level invariance, structured kernel hypotheses and working acquisition loss under ordinary and catalyst-disjoint queries. Their informative and unsuccessful combinations are retained')
    text=text.replace('The blend is not a validated universal recommendation or a safety guarantee; its prior can still carry an unreliable global component.',
        'The predictor-matched extension improves some equal-budget comparisons but reverses its advantage between ordinary and catalyst-disjoint queries. Its stronger isotropic control and non-improving covariance-averaged candidate further show that a coherent working-risk calculation does not establish algorithmic superiority. The blend prior can also carry an unreliable global component.')
    text=text.replace('This distinction matters both for deployment and for interpreting a method\'s contribution.',
        'This distinction matters both for deployment and for interpreting a method\'s contribution. The supplementary opened-family budget curves also show that additional labels need not produce a monotonic error reduction for a fixed, history-tuned local model.')
    cstart=text.index('\\section{Conclusion}')
    cend=text.index('\\input{references.tex}',cstart)
    text=text[:cstart]+r'''\section{Conclusion}
Target-source labels supply an output level and, with informative features, input--response correspondence. Established query-directed acquisition increases the value of a fixed label budget on methanol STY and composition-enriched OCM. Chemistry useful for choosing measurements can be separated from chemistry used by a response predictor. The earlier frozen publication application retains a 12.0\% acquisition point improvement for the composition blend. A later catalyst-disjoint stress test supplies a 23.4\% point improvement for an unchanged local RBF under matched-kernel acquisition, while ordinary random queries reverse that acquisition's advantage. Strong local baselines, matched tuning controls and unsuccessful learned or covariance-averaged policies delimit the extra value of historical response structure and new working assumptions. The supported contribution concerns conditional calibration information and the task-dependent value of measurement locations. Cross-laboratory prospective validation and a consistently superior new algorithm remain empirical questions.

\section*{Data and code availability}
The tenth numbered complete package contains the current manuscript and SI, four additional scientific figures, code, frozen historical kernel choices, every measured result and execution/audit log, and catalyst/support/query identifier ledgers. Its immutable \texttt{reference\_step9} preserves the complete preceding release, including all previous inputs, original full manuscripts, models, source/measurement provenance and earlier raw prediction arrays. The current history comprises twenty-five seed/fold comparisons. It reuses the preceding backbones and adds actual kernel validation, nested support selection, prediction and audit, with no new backbone fits. All three current applications use the same previously evaluated author dataset; none is counted as a second independent publication. The author CADS measurements and paper are attributed to Nishimura et al.\ \cite{nishimura2023new}; upstream terms remain applicable. Full electronic tables provide all twelve acquisitions crossed with seven predictors, both representations and all budgets. A public integrated archive DOI and a future untouched experiment would support stronger dissemination and confirmation. Local hash locks are not public preregistration.

'''+text[cend:]
    p.write_text(text)
    p=ROOT/'manuscripts/supplementary_en.tex';text=(original/p.name).read_text()
    text=text.replace('\\input{step9_supplementary.tex}','\\input{step9_supplementary.tex}\n\\input{step10_supplementary.tex}')
    p.write_text(text)
    p=ROOT/'manuscripts/step9_methods_main.tex';text=(original/p.name).read_text().replace('A frozen application to new author-reported measurements','The earlier frozen application to new author-reported measurements');p.write_text(text)
    p=ROOT/'manuscripts/references.tex';text=(original/p.name).read_text()
    text=text.replace(r'\begin{thebibliography}{12}\small',r'\begin{thebibliography}{99}\small\setlength{\itemsep}{1pt}')
    text=text.replace(r'\bibitem{gpml} C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. MIT Press (2006), Chapter 2, Section 2.7. \url{https://gaussianprocess.org/gpml/chapters/RW2.pdf}.',
        r'\bibitem{gpml} C. E. Rasmussen and C. K. I. Williams. Gaussian Processes for Machine Learning. MIT Press (2006), Chapters 2 and 4. \url{https://gaussianprocess.org/gpml/}.')
    additional=r'''
\bibitem{takeno2025} S. Takeno et al. Distributionally Robust Active Learning for Gaussian Process Regression. ICML, PMLR 267, 58339--58358 (2025). \url{https://proceedings.mlr.press/v267/takeno25a.html}.
\bibitem{tang2026} R. Tang, S. J. Sloman and S. Kaski. Representative, Informative, and De-Amplifying: Requirements for Robust Bayesian Active Learning under Model Misspecification. AISTATS, PMLR 300, 3016--3024 (2026). \url{https://proceedings.mlr.press/v300/tang26d.html}.
'''
    text=text.replace('\\end{thebibliography}',additional+'\n\\end{thebibliography}')
    p.write_text(text)


if __name__=='__main__':
    main()
