"""Build the complete paper/SI and Chinese note from measured Step7 outputs."""
import os
for k in ('OPENBLAS_NUM_THREADS','OMP_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='1'
from pathlib import Path
import json,subprocess
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import fitz
from reportlab.lib import colors
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import SimpleDocTemplate,Paragraph,Spacer,Table,TableStyle,PageBreak

ROOT=Path(__file__).resolve().parents[1]
ENDPOINTS=['methanol_conversion','methanol_selectivity','methanol_sty','ocm_f0','ocm_f1']
NAMES=['Methanol conversion','Methanol selectivity','Methanol STY','OCM operating (F0)','OCM composition (F1)']
CNAMES=['甲醇转化率','甲醇选择性','甲醇STY','OCM操作特征 F0','OCM含组成特征 F1']
POLICIES=['random','temperature_cover','feature_cover','linear_vopt','linear_pool_vopt']
PNAMES=['Random','Temperature cover','Feature cover','Query V','Pool V']
METHODS=['base','target_mean','offset_mean','affine','eb','local_ridge','temperature_ridge','residual_ridge','adaptive_blend','local_ridge_retuned','residual_ridge_retuned','adaptive_blend_retuned','local_rbf_retuned']
MNAMES=['Global','Support mean','Offset','Affine','Empirical Bayes','Frozen local ridge','Temperature ridge','Frozen residual ridge','Frozen blend','Retuned local ridge','Retuned residual ridge','Retuned blend','Retuned local RBF']


def write(path,text):Path(path).write_text(text,encoding='utf-8')


def sf(fig,name):
    fig.savefig(ROOT/'figures'/f'{name}.png',dpi=200,bbox_inches='tight',facecolor='white')
    fig.savefig(ROOT/'figures'/f'{name}.pdf',bbox_inches='tight',facecolor='white');plt.close(fig)


def figures(s,f,p,geometry):
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42})
    f=f.set_index('endpoint').loc[ENDPOINTS]
    fig,ax=plt.subplots(figsize=(9,4.6),layout='constrained')
    for i,(_,row) in enumerate(f.iterrows()):
        scale=100/row.reference_mae;x=row.delta_mae*scale;lo=row.family_ci_low*scale;hi=row.family_ci_high*scale
        color='#247565' if hi<0 else '#526b91'
        ax.errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color=color,capsize=5,lw=2)
        ax.text(hi+1.4,i,f'n={int(row.n_sources)}',va='center',fontsize=9,color='#657483')
    ax.axvline(0,color='#8b96a1',ls='--',lw=1);ax.set_yticks(range(5),NAMES);ax.invert_yaxis();ax.set_xlim(-48,29)
    ax.set_xlabel('MAE difference / observed random-design MAE (%)');ax.set_title('Five labels | unchanged frozen blend\nFive-comparison adjusted source-group intervals',fontsize=12)
    ax.grid(axis='x',alpha=.15);sf(fig,'step7_01_primary')
    fig,axes=plt.subplots(2,3,figsize=(11.2,7.3),layout='constrained')
    for endpoint,name,ax in zip(ENDPOINTS,NAMES,axes.flat):
        for policy,label,color in [('random','Random','#8b96a1'),('linear_vopt','Query V','#247565'),('linear_pool_vopt','Pool V','#536fab'),('feature_cover','Feature cover','#bf9055')]:
            g=s[(s.endpoint==endpoint)&(s.scenario=='random')&(s.policy==policy)&(s.method=='adaptive_blend')].sort_values('k')
            ax.plot(g.k,g.mae,'o-',label=label,color=color,lw=1.7,ms=4)
        ax.set_title(name,fontsize=10.5);ax.set_xticks([1,2,3,5,10]);ax.set_xlabel('Labels');ax.set_ylabel('MAE (native STY unit)' if endpoint=='methanol_sty' else 'MAE (pp)');ax.grid(axis='y',alpha=.17)
    axes.flat[-1].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labels,loc='center',frameon=False,fontsize=11)
    sf(fig,'step7_02_budgets')
    fig,axes=plt.subplots(1,2,figsize=(11,4.7),layout='constrained');matrix=[]
    for endpoint in ENDPOINTS:
        g=s[(s.endpoint==endpoint)&(s.scenario=='random')&(s.k==5)&(s.method=='adaptive_blend')].set_index('policy').loc[POLICIES]
        matrix.append(100*g.mae.to_numpy()/g.loc['random','mae'])
    matrix=np.array(matrix);im=axes[0].imshow(matrix,cmap='RdYlGn_r',vmin=75,vmax=125,aspect='auto')
    axes[0].set_yticks(range(5),NAMES,fontsize=9.5);axes[0].set_xticks(range(5),['Random','Temp.','Feature','Query V','Pool V'],rotation=30,ha='right',fontsize=9.5)
    for i in range(5):
        for j in range(5):axes[0].text(j,i,f'{matrix[i,j]:.1f}',ha='center',va='center',fontsize=9)
    axes[0].set_title('Frozen blend MAE\nRandom = 100 in each panel',fontsize=11)
    fig.colorbar(im,ax=axes[0],shrink=.75,label='MAE index')
    pp=p[(p.scenario=='random')&(p.policy=='linear_vopt')].set_index('endpoint').loc[ENDPOINTS]
    for i,(_,row) in enumerate(pp.iterrows()):
        scale=100/row.aligned_mae;x=row.delta_mae*scale;lo=row.ci_low*scale;hi=row.ci_high*scale
        axes[1].errorbar(x,i,xerr=[[x-lo],[hi-x]],fmt='o',color='#516b91',capsize=4)
    axes[1].axvline(0,color='#8b96a1',ls='--');axes[1].set_yticks(range(5),NAMES,fontsize=9.5);axes[1].invert_yaxis();axes[1].set_xlabel('Shuffle minus aligned MAE / aligned MAE (%)');axes[1].set_title('Same five labels; correspondence shuffled\nExploratory nominal source-group intervals',fontsize=11);axes[1].grid(axis='x',alpha=.15)
    sf(fig,'step7_03_information')
    g=geometry[(geometry.scenario=='random')&(geometry.k==5)];fig,ax=plt.subplots(figsize=(7.2,4.5),layout='constrained')
    for j,(design,name,color) in enumerate([('operating_v_same_alpha','Operating geometry','#8b9fba'),('composition_v_same_alpha','Composition geometry','#2b7b68')]):
        rows=g[g.design==design].set_index('predictor').loc[['ocm_f0','ocm_f1']]
        bars=ax.bar(np.arange(2)+(j-.5)*.33,rows.mae__adaptive_blend,width=.32,label=name,color=color)
        ax.bar_label(bars,fmt='%.3f',padding=4,fontsize=10)
    ax.set_xticks([0,1],['F0 predictor','F1 predictor']);ax.set_ylabel('Source-equal MAE (pp)');ax.set_ylim(0,4.6);ax.set_title('Post-primary OCM diagnostic | five labels\nSame alpha schedule and same pool/query rows',fontsize=11);ax.legend(loc='upper right',frameon=False,fontsize=9.5);sf(fig,'step7_04_geometry')


def make_tables(s,f,p,contrasts,geometry):
    dest=ROOT/'manuscripts/tables';dest.mkdir(exist_ok=True)
    lines=[r'\begin{tabular}{lrrrrl}\toprule Panel & $n$ & Random & Query V & Gain (\%) & Adjusted $\Delta$ interval\\\midrule']
    f=f.set_index('endpoint')
    for endpoint,name in zip(ENDPOINTS,['Conversion','Selectivity','STY','OCM F0','OCM F1']):
        row=f.loc[endpoint];lines.append(f'{name} & {int(row.n_sources)} & {row.reference_mae:.3f} & {row.mae:.3f} & {row.relative_reduction_percent:.1f} & [{row.family_ci_low:.3f}, {row.family_ci_high:.3f}]\\\\')
    lines.append(r'\bottomrule\end{tabular}');write(dest/'step7_primary.tex','\n'.join(lines)+'\n')
    lines=[r'\begin{tabular}{lrrrrr}\toprule Panel & R & T & F & V & P\\\midrule']
    for endpoint,name in zip(ENDPOINTS,['Conversion','Selectivity','STY','OCM F0','OCM F1']):
        rows=s[(s.endpoint==endpoint)&(s.scenario=='random')&(s.k==5)&(s.method=='adaptive_blend')].set_index('policy').loc[POLICIES]
        lines.append(name+' & '+' & '.join(f'{v:.3f}' for v in rows.mae)+r'\\')
    lines.append(r'\bottomrule\end{tabular}');write(dest/'step7_acquisition.tex','\n'.join(lines)+'\n')
    factorial=[]
    for endpoint,name in zip(ENDPOINTS,NAMES):
        rows=s[(s.endpoint==endpoint)&(s.scenario=='random')&(s.k==5)].pivot(index='method',columns='policy',values='mae').loc[METHODS,POLICIES]
        factorial.extend([r'\begin{table}[htbp]\centering\small',r'\caption{'+name+r': all five-label random-query factors. Native STY units for STY; percentage points otherwise.}',r'\begin{tabular}{lrrrrr}\toprule Predictor & R & T & F & V & P\\\midrule'])
        for method,label in zip(METHODS,MNAMES):factorial.append(label+' & '+' & '.join(f'{v:.3f}' for v in rows.loc[method])+r'\\')
        factorial.append(r'\bottomrule\end{tabular}\par\smallskip\footnotesize R: random; T: temperature cover; F: feature cover; V: query V-optimal; P: pool V-optimal. Supports are common across predictors within a policy.\end{table}\FloatBarrier')
    write(dest/'step7_factorial_si.tex','\n'.join(factorial)+'\n')


def manuscript(s,f,p,contrasts,geometry,v,a):
    dest=ROOT/'manuscripts';source=ROOT/'reference_step6/manuscripts';main=(source/'manuscript_en.tex').read_text();si=(source/'supplementary_en.tex').read_text()
    abstract=r'''Few target-source measurements can recover catalytic prediction accuracy, but their value cannot be attributed to label count alone. We separate response level, input--label correspondence, historical response reuse and acquisition geometry under publication shift. Source-disjoint analyses cover methanol conversion, selectivity and space-time yield (STY), plus operating-only and composition-enriched oxidative coupling of methane (OCM): five panels from four outcomes in two reaction systems. A retrospective extension crosses five acquisition rules with thirteen predictors, retaining fixed pools, queries and nested label budgets. At five labels, established linear V-optimal acquisition reduces an unchanged blend's source-equal mean absolute error from 62.608 to 50.274 in native STY units and from 3.081 to 2.708 percentage points for composition-enriched OCM; both gains survive five-comparison adjustment. Conversion and selectivity show favorable point estimates but inconclusive adjusted intervals; operating-only OCM shows no gain. Previously evaluated additional OCM publications give a reduction from 3.552 to 3.140 percentage points. Label shuffling preserves response means yet degrades informative feature-dependent fits. A matched OCM follow-up separates acquisition representation from prediction representation. More elaborate historical covariance transfer does not improve strong controls. Labels support source-conditioned prediction, and informative locations can increase their value; neither establishes universal zero-label generalization or a superior new algorithm.'''
    start=main.index(r'\begin{abstract}')+len(r'\begin{abstract}');end=main.index(r'\end{abstract}');main=main[:start]+'\n'+abstract+'\n'+main[end:]
    main=main.replace('target information, residual uncertainty and acquisition geometry','what target labels contribute and where to acquire them')
    main=main.replace('A subsequent candidate tests whether learning source-contrast residual covariance and aligning prediction with acquisition yields a stronger method.',
        'A five-panel extension tests the portability of acquisition gains, with local, kernel and historical-response controls. More complex transfer candidates are evaluated as secondary mechanism checks.')
    mstart=main.index(r'\subsection{Learning a correction covariance');mend=main.index(r'\section{Results}',mstart)
    short=r'''\subsection{Secondary uncertainty-transfer candidates}
Component-aware acquisition and a subsequent residual-covariance candidate test whether historically learned uncertainty adds predictive value. The latter is a finite-rank Gaussian-process/random-coefficient model with a free source intercept, historically learned coefficient covariance, and aligned posterior-variance acquisition. These ingredients are established \cite{gpml,alpaca,gpmllearn,scaml}. Source-disjoint response features and a further nested cross-fit exclude the entire pseudo-validation fold during covariance learning. Five covariance alternatives and strong policy-specific controls are evaluated, with four locked candidate contrasts per cohort. Complete derivations, parameter-matched ablations and all outcomes are retained in Supplementary Section 13. Working covariance is not measured laboratory noise or validated predictive uncertainty \cite{iwata}.

\subsection{Five-panel test of acquisition geometry}
The new retrospective extension uses every one of the 25 inherited outer fits per panel, with the original query populations and budgets $1,2,3,5,10$. Nine frozen predictors are crossed with random acquisition, temperature coverage, full-feature coverage, query-directed linear V-optimal design and a pool-only counterpart. Four additional controls re-tune local and residual ridge, blending, and local RBF regression by policy and budget on historical pseudo-sources only. Historical source-disjoint OOF predictions are reused with their matching inner training preprocessors; 125 outer frozen-configuration models are actually re-fitted. This extension does not train 125 new nested searches or choose configurations from outer queries.

For a working feature row $\psi(x)=[10,z(x)^\top]^\top$, the standard linear design objective is
\begin{equation}
J_Q(A)=\frac{1}{n_Q}\operatorname{tr}\left[F_Q\left(I-F_A^\top(F_AF_A^\top+\alpha I)^{-1}F_A\right)F_Q^\top\right].
\label{eq:crossdesign}
\end{equation}
Each greedy addition minimizes $J_Q$. Its unit coefficient prior, intercept prior variance 100 and inherited ridge regularizer form a working surrogate, not a claim of calibrated physical uncertainty. Prediction retains the original unpenalized intercept. Query V uses known query covariates and is transductive; pool V replaces $F_Q$ with the candidate pool and does not use query covariates. Coverage rules choose the point nearest the pool center first, then maximize distance from selected points. Every selector is label-blind. Random uses 50 nested orders; each deterministic rule uses one. All predictors share a policy's supports. V-optimal design is established \cite{yubi,fontaine}; these are explicit standard-design implementations, not complete reproductions of those papers' algorithms.

The primary family is the frozen blend's query-V-minus-random MAE at five labels and random-query allocation in each of the five panels. Leakage-group bootstrap uses 20,000 draws, averaging support draws and seeds before sources. Nominal 95\% and five-member Bonferroni-adjusted 99\% percentile intervals are reported. The panels share records and sources, so they are not independent replications; Bonferroni adjustment does not require independence. All other budgets, rules, control increments, stress panels and label shuffles are exploratory. This local protocol lock precedes current outer evaluation, but the same corpora have already informed earlier method development.

'''
    main=main[:mstart]+short+main[mend:]
    rstart=main.index(r'\subsection{Cross-fitted residual covariance');rend=main.index(r'\section{Discussion}',rstart)
    results=r'''\subsection{Standard acquisition gains extend across tasks, with limits}
At the same five-label budget and with the same frozen blend, query V-optimal acquisition reduces STY MAE by 19.7\% and composition-enriched OCM MAE by 12.1\% (Table~\ref{tab:crossprimary}; Fig.~\ref{fig:crossprimary}). Their adjusted differences are $-12.335$ [$-25.965,-2.874$] in native STY units and $-0.372$ [$-0.746,-0.056$]\pp, respectively. The conversion point estimate improves 13.1\%, but its adjusted interval crosses zero; the nominal interval is favorable. Selectivity improves 7.8\% in point estimate with both nominal and adjusted intervals crossing zero. Operating-only OCM has a 3.0\% higher point estimate, also inconclusive. Four favorable points out of five therefore do not establish universal acquisition superiority.

\begin{table}[htbp]\centering\small
\caption{Locked five-panel primary comparison at five labels: unchanged frozen blend, random-query allocation. Gain is relative MAE reduction; negative values mean higher error. MAE and difference intervals use percentage points except STY, which uses \styunit. Intervals are adjusted across the five comparisons.}\label{tab:crossprimary}
\input{tables/step7_primary.tex}\end{table}
\plotfigure{step7_01_primary}{Standard query-directed V-optimal acquisition minus random acquisition. Intervals are the primary adjusted native-unit difference intervals rescaled by each panel's fixed observed random MAE for display; they are not a bootstrap ratio interval or a pooled reaction effect. Negative values favor query V. Source counts and representation dependence constrain the claim.}{fig:crossprimary}

The broader controls preserve this interpretation. At five labels, re-tuned blending gives STY MAE 49.852 under query V versus 62.613 under random acquisition. For OCM F1 it gives 2.696 versus 3.073\pp. On the same query-V supports, ordinary local ridge and local RBF give STY MAEs 51.293 and 51.547, and OCM F1 values 2.823 and 2.851\pp. Most blend increments over these local controls remain inconclusive in the other panels. Full-feature coverage and pool-only design can help, but neither rule is a universal winner. Pool-only selectivity has a lower point estimate than query V, and their paired interval crosses zero. The full factorial and budget curves are in Supplementary Section 14.

High-temperature panels also differ. Historical conversion and selectivity show exploratory query-V improvements on 23 sources, whereas STY's 21-source interval crosses zero. The historical OCM high-temperature set contains only five sources and does not show a reliable gain. These stress outcomes share query populations across acquisition rules; they are not pooled with random-query data or additional-corpus results.

\subsection{Labels encode correspondence, and representation also guides selection}
Under query V, five-label correspondence shuffling leaves the same label multiset and mean but changes the frozen blend's STY MAE from 50.274 to 80.945, and OCM F1 MAE from 2.708 to 3.548\pp. Conversion and selectivity also worsen. For OCM F0, the change is only $+0.047$\pp\ with a nominal interval crossing zero. These exploratory controls show where feature--response correspondence contributes beyond an output mean; they do not identify a causal chemical mechanism.

After observing the primary outcomes, we added a clearly post-primary OCM diagnostic crossing acquisition geometry and prediction representation. Operating-only and composition-enriched V design use the same inherited F1 regularizer schedule and the same pools/queries; F0 and F1 frozen predictors each see both support sets. With the F1 predictor, operating geometry gives 3.300\pp\ and composition geometry 2.708\pp; the nominal paired change is $-0.592$ [$-1.347,-0.020$]\pp. With the F0 predictor, these values are 3.821 and 3.370\pp, a nominal change of $-0.452$ [$-0.982,-0.010$]\pp. Composition features can thus carry useful selection information even for a predictor that does not consume those features. This is exploratory evidence from 22 sources, not an adjusted primary result or proof that composition is necessary in every reaction. STY demonstrates a useful operating-feature acquisition case.

\subsection{Complex uncertainty transfer remains a boundary result}
The preceding residual-covariance candidate does not provide a stronger algorithm. Its historical/additional five-label MAEs are 3.287/3.903\pp\ versus 2.696/3.149\pp\ for re-tuned blending with standard linear design. The four-comparison adjusted candidate increments are positive in both cohorts. Cross-fitted covariance improves a small matched in-sample-covariance contrast, but does not reverse overall inferiority. Working response variance also differs from observed error. Complete candidate results and unfavorable alternatives remain in Supplementary Section 13; their coherent formulas are not treated as performance innovation.

'''
    main=main[:rstart]+results+main[rend:]
    main=main.replace('Standard design improves the frozen blend.','Standard design improves the frozen blend on STY and composition-enriched OCM, with inconclusive adjusted conversion/selectivity results and no operating-only OCM gain.')
    main=main.replace('only explicitly defined four-contrast candidate families receive adjustment, separately within each cohort.',
        'the earlier four-contrast candidate families are adjusted separately within each cohort, and the current five-panel acquisition primary family also receives adjustment.')
    cstart=main.index(r'\section{Conclusion}');cend=main.index(r'\input{references.tex}',cstart)
    conclusion=r'''\section{Conclusion}
Target-source labels supply an output level and, in informative representations, input--response correspondence. Acquisition geometry changes how much those labels contribute at the same budget. Standard query-directed design improves STY and composition-enriched OCM under a fixed predictor, while other panels set clear limits. An exploratory representation crossing also locates information useful for selection separately from information consumed by prediction. Historical response transfer contributes an additional, context-dependent increment; complex covariance repairs do not establish superiority. These findings support a bounded scientific claim about target information and measurement placement, not universal zero-label generalization or a new optimal-design algorithm.

\section*{Data and code availability}
The seventh numbered package contains cleaned multi-task input snapshots, all 125 inherited parameter/OOF/split records needed by this extension, current support/query ledgers, every current float64 query prediction, measured summary tables, code and complete manuscript/figure sources. Its entry point verifies released data and actually re-fits current outer models in a fresh directory; inherited OOF responses are explicitly reused for historical tuning. Earlier complete releases retain upstream attributions, additional-source inclusion/bibliography ledgers, external prediction exports and the full older candidate stages. Their identifiers and cryptographic hashes are recorded in the current package. A final public repository or archive DOI should integrate those stages; a local stage lock is not independent preregistration.

'''
    main=main[:cstart]+conclusion+main[cend:];write(dest/'manuscript_en.tex',main)
    si=si.replace('The current covariance-stage package independently contains all experiments in Section 13, including their inputs, predecessor settings and prediction exports.',
        'The sixth release independently contains all experiments in Section 13, including its inputs, predecessor settings and prediction exports. The current seventh release independently reproduces Section 14; earlier full releases remain separate, with exact identifiers and hashes.')
    si=si.replace('The covariance-stage reproduction entry point re-fits this stage; it does not claim to newly re-fit all earlier multi-task studies. Full earlier records are in the preceding packages. Every current cell, support/query ledger, covariance record, fitted parameter, prediction export, protocol lock and figure source is included.',
        'The sixth release\'s covariance entry point re-fits Section 13; it does not claim to newly re-fit all earlier multi-task studies. Full covariance records and arrays remain in that sixth release. The seventh release preserves the covariance summaries and figures but re-fits the separate cross-task acquisition experiment in Section 14.')
    addition=r'''\section{Cross-task standard acquisition extension}
\subsection{Scope, chronology and fixed populations}
This seventh extension evaluates five panels from four outcomes in two reaction systems, not five independent reaction datasets. Conversion and selectivity share the same paired records; twenty publication identifiers overlap the STY collection; OCM F0/F1 use the same records. These dependencies remain relevant when interpreting breadth. The current protocol is timestamped and hashed before current outer evaluation, but all input corpora were already inspected in earlier development. All new results are retrospective. Only the five-label frozen-blend query-V-minus-random comparison across the five random-query panels is primary. The parameter-matched OCM geometry crossing was added after primary results and is explicitly exploratory.

Every outer seed/fold and backbone configuration is inherited verbatim from the preceding full-study benchmark. Current outer models are re-fitted and compared with the original zero-label predictions. Historical source-disjoint OOF responses are reused for policy-specific control tuning. Inner preprocessors are reconstructed using the same three historical training-group partitions; no pseudo-validation group fits its own feature preprocessor. There is no new inner backbone fit or new global-configuration search in this extension. OOF model configuration was selected within the complete outer historical-training set in the earlier benchmark; historical control scores remain optimization diagnostics, not independent test results. Outer query groups are excluded from every such choice.

Random queries, high-temperature queries and reported-stratum holdouts use the original label-blind definitions. Eligibility is unchanged: at least ten pool records, at least five query records. Pool IDs are sorted before selection. Deterministic acquisitions are evaluated once, rather than copied fifty times as artificial replicates. Random orders run fifty times. Every policy produces one nested sequence to budget ten; intermediate steps use the original local-ridge alpha of the next reported budget. Different target outcomes can have different frozen alpha schedules, even when they share feature rows.

\subsection{Five acquisitions and thirteen predictors}
Temperature coverage uses the first numeric feature; full-feature coverage uses all allowed standardized/imputed numeric and encoded-composition features. The first row is nearest the pool mean and later rows maximize their minimum squared distance to the selected set. This is a conventional coverage heuristic. Query and pool V-optimal acquisition use a unit linear coefficient prior on $[10,z]$ and the inherited random-design local alpha. Query V minimizes mean working posterior variance on query covariates; pool V minimizes it over all candidate-pool covariates, including candidates that may later be observed. Neither reads outcomes. The finite intercept prior used for acquisition differs from the exact unpenalized intercept used for prediction, so working variance is only a design surrogate.

Nine original predictors are frozen: global, support mean, residual offset, affine, empirical Bayes, local ridge, temperature ridge, residual ridge and shrinkage blend. Four stronger controls re-tune local ridge, residual ridge, blending and local RBF by policy/budget. Ridge alphas are $\{0.1,1,10,100\}$. RBF alpha and gamma both use $\{0.01,0.1,1\}$ and $\{0.1,1,10\}$, respectively; squared distances divide by feature dimension. Kernel ridge centers its support kernel and response and restores the support mean. After local/residual alpha selection, blending evaluates five prior weights times five shrinkages, using the inherited response scale and support-only analytic LOO. Each candidate's MAE is averaged over historical support draws, then pseudo-sources. Random pseudo-supports repeat ten times; deterministic policies once. All control grids and acquired rows are common across policies. An ordinary kernel/local fit is a strong comparator; a lower point estimate is not selected using outer query outcomes.

\subsection{Primary family, units and aggregation}
Each error cell averages query rows, then repeated support draws are averaged within source/seed, then seeds within source, then sources receive equal weight. A leakage group may contain more than one source; cluster bootstrap retains those source weights. Twenty thousand group resamples produce nominal 95\% intervals. Five-panel primary tails are 0.005 and 0.995, the approximate Bonferroni-adjusted 99\% percentile interval. Adjustment accommodates endpoint dependence but does not include training-fit uncertainty or remove retrospective selection. No folds, random draws, or shuffled-label draws are counted as independent publications. STY uses its native productivity unit, while the other outcomes use percentage points; raw errors are not pooled across these units.

\begin{table}[htbp]\centering\small\caption{Five-panel adjusted primary results. Same interpretation and units as the main primary table.}\input{tables/step7_primary.tex}\end{table}
\begin{table}[htbp]\centering\small\caption{Frozen blend across five acquisitions at five labels. R: random; T: temperature coverage; F: feature coverage; V: query V; P: pool V. STY has native productivity units; other rows are percentage points.}\input{tables/step7_acquisition.tex}\end{table}
\FloatBarrier\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{../figures/step7_02_budgets.pdf}\caption{Budget curves for the unchanged frozen blend on five random-query panels. All rules share each panel's query population. The four displayed rules are representative; all five acquisitions and thirteen predictors are available in the numerical tables.}\end{figure}\FloatBarrier

'''
    addition+=r'\input{tables/step7_factorial_si.tex}'+'\n'
    addition+=r'''\subsection{Label-correspondence diagnostic}
At budget five, the first random order and each deterministic support are used for twenty outcome permutations per source/scenario/seed. The input rows, global predictions, support outcomes as a multiset, and exact support mean remain the same; only input--label correspondence changes. The frozen blend is re-computed, including its support-only blend weight. Query labels enter scoring only. These are exploratory controls, not repeated training datasets or a physical perturbation experiment. Permutation errors average shuffles and seeds before source-group intervals.

\begin{table}[htbp]\centering\small\caption{Correspondence shuffling under query V: frozen blend, five labels. Nominal source-group intervals are exploratory.}\begin{tabular}{lrrl}\toprule Panel & Aligned & Shuffled & Shuffled minus aligned, 95\% CI\\\midrule
'''
    for endpoint,name in zip(ENDPOINTS,['Conversion','Selectivity','STY','OCM F0','OCM F1']):
        row=p[(p.endpoint==endpoint)&(p.scenario=='random')&(p.policy=='linear_vopt')].iloc[0]
        addition+=f'{name} & {row.aligned_mae:.3f} & {row.permuted_mae:.3f} & {row.delta_mae:.3f} [{row.ci_low:.3f}, {row.ci_high:.3f}]\\\\\n'
    addition+=r'''\bottomrule\end{tabular}\end{table}
\FloatBarrier\begin{figure}[htbp]\centering\includegraphics[width=\linewidth]{../figures/step7_03_information.pdf}\caption{Left: all acquisition means for the frozen blend, normalized to each panel's observed random-design MAE. Right: outcome-correspondence shuffling effect, nominal native-unit intervals rescaled by the fixed observed aligned MAE. Normalization is for display, not a pooled or bootstrap ratio estimand.}\end{figure}\FloatBarrier

\subsection{Post-primary representation crossing}
The OCM follow-up fixes the F1 alpha schedule for both operating-only (six-feature) and full composition-enriched V geometry. It evaluates each acquired sequence with both already frozen F0 and F1 predictors. Numeric preprocessors are verified equal between paired representations; outcomes, pool rows and queries are identical. The full-geometry orders exactly match the original F1 orders. The operating-only orders in this follow-up can differ from the primary F0 orders because the regularizer is now fixed to F1, rather than independently inherited from F0. A direct comparison of the primary F0 and F1 numbers would mix representation, backbone/adapter choices and acquisition regularization; this crossing removes the acquisition-alpha difference for its own conditional contrasts.

\begin{figure}[htbp]\centering\includegraphics[width=.85\linewidth]{../figures/step7_04_geometry.pdf}\caption{Post-primary exploratory crossing of selection geometry and prediction representation. Same F1 alpha, five supports, 22 OCM sources. Geometry-specific nominal intervals and all budgets are released. No new training or parameter selection occurs. Composition used for acquisition is not automatically an input to the F0 predictor.}\end{figure}\FloatBarrier

\subsection{Independent verification and reproduction scope}
'''
    addition+=f'The current computation contains {v["parts"]} outer fits, {v["cells"]:,} error cells, {v["designs"]:,} nested orders, and {v["row_values"]:,} scalar row predictions across 65 combinations per panel. Exactly 125 new outer backbones and zero new inner backbones are trained. Original random-design MAEs reproduce across {v["archived_random_cells"]:,} cells and nine methods, with maximum discrepancy {v["archived_random_mae_maximum_difference"]:.3g}. The larger STY scale explains why this roundoff is not identical to the percentage-point panels. OCM frozen query/random orders and original four-method errors independently match the sixth release.\n\n'
    addition+=f'Independent audits re-aggregate {a["row_values_reaggregated"]:,} scalar predictions and reconstruct {a["row_values_recomputed"]:,} from support rows. All {a["poisoned_label_orders_recomputed"]:,} selection orders remain unchanged after replacing every target response by values between $-10^9$ and $10^9$. Five hundred outer/inner group-separation checks pass. Mathematical verification compares all twenty greedy steps of query/pool design with a direct primal posterior and checks exact ridge prediction/LOO against independent scikit-learn fits. The post-primary crossing adds 2,700 cells and 172,800 scalar predictions, with an independent read-only audit.\n\n'
    addition+=r'''The seventh release includes lossless float64 NPZ predictions, acquired/query ID ledgers, historical control scores, every parameter, input snapshots, split ledgers, predecessor references and a checksummed manifest. Its verify mode reads the release and writes audit reports only into a temporary directory. Smoke mode re-fits one OCM F1 outer fold in an empty external directory; full mode re-fits all 125 current outer models, re-tunes the current controls from inherited historical OOF records and repeats the post-primary geometry diagnostic. A completed cache is never counted as a fresh refit. The smoke check is a portability check, not a second full 125-fit independent repetition. Earlier Section 1--13 studies retain their complete raw arrays in the prior numbered releases, rather than being claimed as newly re-run by the current entry point.

'''
    si=si.replace(r'\input{references.tex}',addition+'\n'+r'\input{references.tex}',1);write(dest/'supplementary_en.tex',si)
    for name in ['manuscript_en','supplementary_en']:
        for suffix in ['.aux','.out','.log']:(dest/(name+suffix)).unlink(missing_ok=True)
        for _ in range(2):
            result=subprocess.run(['pdflatex','-interaction=nonstopmode','-halt-on-error',name+'.tex'],cwd=dest,capture_output=True,text=True)
            if result.returncode:raise RuntimeError(result.stdout[-5000:])
        print(name,len(fitz.open(dest/(name+'.pdf'))),'pages',flush=True)


def chinese(f,v,a):
    fontdir=ROOT/'build_fonts';fontdir.mkdir(exist_ok=True);font=fontdir/'paper_cjk.ttf';font.write_bytes(fitz.Font('cjk').buffer);pdfmetrics.registerFont(TTFont('ResearchCJK',str(font)))
    body=ParagraphStyle('body',fontName='ResearchCJK',fontSize=10.5,leading=17,spaceAfter=9,wordWrap='CJK',allowWidows=0,allowOrphans=0,textColor=colors.HexColor('#26394c'))
    head=ParagraphStyle('head',parent=body,fontSize=17,leading=25,spaceAfter=14,keepWithNext=True,textColor=colors.HexColor('#133c52'))
    sub=ParagraphStyle('sub',parent=body,fontSize=12,leading=19,spaceBefore=8,spaceAfter=8,keepWithNext=True,textColor=colors.HexColor('#133c52'))
    small=ParagraphStyle('small',parent=body,fontSize=9,leading=14)
    story=[]
    def p(text,style=body):story.append(Paragraph(text,style))
    def page():story.append(PageBreak())
    p('第7步：标签位置的跨任务检验',head)
    p('结论：实证覆盖补强了，收益有明确边界。',sub)
    p('这次实际完成五个面板的同预算选样比较：甲醇转化率、选择性、STY，以及OCM操作特征版和含组成特征版。继续沿用原来源分组、固定查询集和模型参数，避免把更换数据范围或模型带来的变化误算为选样收益。五个面板来自四个响应、两个反应体系，包含共享来源和记录，不是五套独立反应数据。')
    p('同样5个标签、同一个冻结混合模型',sub)
    table=[[Paragraph(x,small) for x in ['面板','随机选样','查询V选样','误差降低','调整后判断']]]
    for endpoint,name in zip(ENDPOINTS,CNAMES):
        row=f[f.endpoint==endpoint].iloc[0];judgment='支持降低' if row.family_ci_high<0 else '区间跨零'
        table.append([Paragraph(name,small),Paragraph(f'{row.reference_mae:.3f}',small),Paragraph(f'{row.mae:.3f}',small),Paragraph(f'{row.relative_reduction_percent:.1f}%',small),Paragraph(judgment,small)])
    t=Table(table,colWidths=[140,72,72,68,95],repeatRows=1);t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor('#e7eff3')),('VALIGN',(0,0),(-1,-1),'TOP'),('LEFTPADDING',(0,0),(-1,-1),7),('RIGHTPADDING',(0,0),(-1,-1),7),('TOPPADDING',(0,0),(-1,-1),8),('BOTTOMPADDING',(0,0),(-1,-1),8),('LINEBELOW',(0,0),(-1,0),.6,colors.HexColor('#b5c7d0'))]));story.append(t);story.append(Spacer(1,10))
    p('STY采用mg MeOH h-1 gcat-1，其余为百分点，不能把原始误差混在一起平均。STY的19.7%和OCM F1的12.1%收益通过了五项主比较调整。转化率的普通95%区间有利，但调整后跨零；选择性同样不能下稳定改善的结论。OCM F0的误差点估计增加约3%，区间跨零。')
    p('强对照没有被省略',sub)
    p('五种选样包括随机、温度覆盖、全特征覆盖、查询集V-optimal及不使用查询特征的候选池V-optimal。十三个预测器包含原九种策略，以及按选样与预算重新调参的局部岭回归、残差岭回归、混合模型和局部RBF核回归。总计65个组合，不等于65种原创算法。')
    page();p('标签为什么有用？这次回答得更具体',head)
    p('1. 标签提供来源内响应水平和输入对应关系',sub)
    p('标签改变了预测条件：模型已经测过新来源的若干响应。它可以学习水平，也可以在有信息的特征里学习哪些输入对应哪些输出。这个结论属于有标签条件下的来源内预测，不能写成零标签模型对所有未来来源的泛化更强。')
    p('这次保持5个标签的数值集合与均值，只打乱对应输入：查询V选样下，STY的MAE从50.274升至80.945；OCM F1从2.708升至3.548。OCM F0只从3.768变成3.815，普通区间跨零。标签价值取决于特征里有没有可用的对应信息，数量本身不足以解释全部收益。该打乱分析是探索性机制对照，不是物理催化实验。')
    p('2. 同样数量的标签，位置会影响信息量',sub)
    p('标准查询V选样在STY和OCM F1上显示可信的同预算收益；温度覆盖并不稳定，全特征覆盖或候选池V选样也可能有效。查询V知道未来要预测的输入位置，属于传导式设置；候选池对照帮助判断这种信息是否必需。没有哪一种规则在所有面板都更好。')
    p('3. 组成信息也可能帮助选位置',sub)
    p('在看到主结果后追加一个同正则参数的OCM探索性对照：对同一个F1预测器，仅用操作特征选样得到3.300，加入组成信息选样得到2.708；即便F0预测器本身不读取组成信息，相应选样结果也从3.821降到3.370。这里的普通配对区间略低于零，但未经主家族调整，不能包装为预先锁定的确认性机制。这个对照清楚区分了“用组成帮助选样”和“预测器直接使用组成”。')
    p('上一轮失败候选继续保留',sub)
    p('残差协方差迁移和分量风险选样仍没有超过强标准对照。全文压缩了这部分，补充材料保留原始推导、负结果与消融。不能因为公式更复杂或加入更多方法就声称算法原创性已经成立。')
    page();p('论文主线、一区差距与完成范围',head)
    p('本轮可以写成的科学主张',sub)
    p('在论文来源变化下，少量测量提供响应水平和输入对应关系；同预算标签的位置影响其价值；这个收益与特征表示和预测场景有关。科学贡献是把标签量、信息内容、位置和历史模型增量分开，并用跨任务强对照检验。STY和OCM F1提供正证据，其他面板界定适用范围。')
    p('对SCI一区的理性判断',sub)
    p('这次补齐了一块有价值的实证覆盖，论文比只展示单一OCM选样结果更完整。但标准岭回归、核回归、覆盖和V-optimal都是已有方法；当前没有证据支持“新算法稳定超过成熟强对照”。如果以方法原创性为主要卖点，这个缺口仍在。以科学问题和可验证的信息边界为主线，价值更清楚，但不能仅凭本轮结果判断能否录用一区。')
    p('仍然需要的新证据',sub)
    p('最关键的是没有参与方法设计的新来源或真实新实验批次，并在获取查询响应前冻结预测器、选样规则、预算和主比较。还需要可靠的催化剂制备、材料状态、反应器或批次信息，才能把“论文来源”推进到有物理解释的变化。已有52个新增OCM来源已经用于回顾性开发，不能重新当成未检视的确认集。计算能完成实现、分析和复现，不能代替未取得的物理测量。')
    p('真实完成与核验',sub)
    p(f'{v["parts"]}个外层骨干实际重训；没有新训内层模型，历史OOF预测明确复用；{v["cells"]:,}个误差单元；{v["designs"]:,}条嵌套选样序列；{v["row_values"]:,}个标量行预测。')
    p(f'独立重汇总{a["row_values_reaggregated"]:,}个行预测数值，从支持标签重算{a["row_values_recomputed"]:,}个；把全部目标响应换成极端假值后，{a["poisoned_label_orders_recomputed"]:,}条选样顺序仍不变。另有2700个OCM表示交叉单元及172800个行预测，单独核验。完成一份新目录单拟合复现，不能将其写成第二次全量125拟合。')
    p('包内有完整英文全文、完整补充材料、中文说明、数值结果、全部当前行预测、冻结参数、来源排除记录、绘图代码和复现入口。前序全量阶段的文件范围与标识另有说明，没有将它们冒充成本轮全部重新执行。')
    def footer(canvas,doc):
        canvas.setFont('ResearchCJK',8);canvas.setFillColor(colors.HexColor('#71818b'));canvas.drawString(48,26,'第7步研究记录 | 跨任务实证 | 不作录用保证');canvas.drawRightString(548,26,str(doc.page))
    doc=SimpleDocTemplate(str(ROOT/'manuscripts/第7步_结果与一区差距.pdf'),pagesize=(595.28,841.89),leftMargin=48,rightMargin=47,topMargin=44,bottomMargin=47);doc.build(story,onFirstPage=footer,onLaterPages=footer)
    print('Chinese',len(fitz.open(ROOT/'manuscripts/第7步_结果与一区差距.pdf')),'pages',flush=True)


def main():
    s=pd.read_csv(ROOT/'results/summary.csv');f=pd.read_csv(ROOT/'results/primary_family.csv');p=pd.read_csv(ROOT/'results/label_permutation_summary.csv');c=pd.read_csv(ROOT/'results/paired_contrasts.csv');g=pd.read_csv(ROOT/'results/ocm_geometry/summary.csv')
    v=json.loads((ROOT/'verification/aggregation.json').read_text());a=json.loads((ROOT/'verification/row_audit.json').read_text())
    figures(s,f,p,g);make_tables(s,f,p,c,g);manuscript(s,f,p,c,g,v,a);chinese(f,v,a)


if __name__=='__main__':main()
