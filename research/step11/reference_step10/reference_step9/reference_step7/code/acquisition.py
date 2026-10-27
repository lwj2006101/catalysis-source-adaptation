"""Established, feature-only nested acquisition rules; no outcome argument."""
from __future__ import annotations
import numpy as np

POLICIES=('random','temperature_cover','feature_cover','linear_vopt','linear_pool_vopt')
BUDGETS=(1,2,3,5,10)


def variance_scores(fp,fq,selected,candidates,alpha):
    """Greedy integrated posterior variance under a unit linear working prior.

    A feature of 10 implements an intercept prior variance of 100. Predictors
    use an unpenalized intercept; the acquisition prior is a design surrogate.
    This is standard V-optimal design, not a newly invented algorithm.
    """
    kpq=fp@fq.T;kpp=fp@fp.T;diagonal=np.sum(fq*fq,axis=1).mean();values=[]
    for start in range(0,len(candidates),64):
        batch=candidates[start:start+64]
        takes=np.column_stack([np.broadcast_to(np.asarray(selected,int),(len(batch),len(selected))),batch])
        gram=kpp[takes[:,:,None],takes[:,None,:]];cross=kpq[takes]
        solved=np.linalg.solve(gram+alpha*np.eye(len(selected)+1)[None],cross)
        values.extend((diagonal-np.einsum('nkq,nkq->n',cross,solved)/len(fq)).tolist())
    return np.asarray(values)


def select_order(z_pool,z_query,policy,alpha_by_k,maximum=10):
    """Pool-relative order. Caller sorts row IDs; first minimum breaks ties.

    Intermediate support sizes use the next reported budget's frozen alpha.
    linear_vopt knows query covariates (transductive); every other rule does
    not use the query set. No policy receives target outcomes.
    """
    zp=np.asarray(z_pool,float);zq=np.asarray(z_query,float)
    assert policy in POLICIES[1:] and maximum<=len(zp) and len(zq)>0
    if policy=='linear_pool_vopt':zq=zp
    fp=np.column_stack([np.full(len(zp),10.),zp]);fq=np.column_stack([np.full(len(zq),10.),zq])
    metric=zp[:,:1] if policy=='temperature_cover' else zp
    center=metric.mean(0);distance=((metric-center)**2).sum(1)
    pairwise=((metric[:,None]-metric[None])**2).sum(2)
    selected=[];trace=[]
    for size in range(1,maximum+1):
        candidates=np.array([i for i in range(len(zp)) if i not in selected],int)
        budget=min(int(k) for k in alpha_by_k if int(k)>=size);alpha=alpha_by_k[str(budget)]
        if policy.endswith('_cover'):
            score=distance[candidates] if not selected else -pairwise[np.ix_(candidates,selected)].min(1)
        else:score=variance_scores(fp,fq,selected,candidates,alpha)
        chosen=int(candidates[int(np.argmin(score))]);selected.append(chosen)
        trace.append(dict(k=size,selected_pool_index=chosen,objective=float(score.min()),alpha=float(alpha)))
    return np.asarray(selected),trace
