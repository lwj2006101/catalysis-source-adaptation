"""Acquisition for a fixed free-intercept RBF, with a covariance-error penalty.

The spectral-ball identity is an elementary robust-design result, not a
calibration guarantee. Its application is tested against matched nominal risk.
"""
import numpy as np


def kernel(x, y, gamma):
    x, y = np.asarray(x, float), np.asarray(y, float)
    d = np.maximum(0, (np.square(x).sum(1)[:, None]
                       + np.square(y).sum(1)[None] - 2*x@y.T) / x.shape[1])
    return np.exp(-gamma*d)


def weights(kss, kqs, alpha):
    n = kss.shape[-1]
    inv = np.linalg.solve(kss+alpha*np.eye(n), np.broadcast_to(np.eye(n), kss.shape))
    v = inv.sum(-1); direct = kqs@inv
    return direct + (1-direct.sum(-1))[..., :, None] * (v/v.sum(-1)[..., None])[..., None, :]


def components(kss, kqs, alpha):
    w = weights(kss, kqs, alpha)
    amplification = np.square(w).sum(-1).mean(-1)
    risk = 1+alpha-2*np.einsum('...qk,...qk->...q', w, kqs)
    risk += np.einsum('...qk,...kl,...ql->...q', w, kss+alpha*np.eye(kss.shape[-1]), w)
    if risk.min() < -1e-7:
        raise ArithmeticError('Invalid working variance')
    return np.maximum(0,risk).mean(-1), amplification, w


def select(kp, kpq, alpha, epsilon, maximum=10):
    selected, trace = [], []
    for k in range(1,maximum+1):
        available = np.array([i for i in range(len(kp)) if i not in selected])
        risks, amps = [], []
        for start in range(0,len(available),64):
            batch=available[start:start+64]
            take=np.column_stack([np.broadcast_to(np.asarray(selected,int),(len(batch),len(selected))),batch])
            rr,aa,_=components(kp[take[:,:,None],take[:,None,:]],kpq[take].transpose(0,2,1),alpha)
            risks.extend(rr); amps.extend(aa)
        risks,amps=np.asarray(risks),np.asarray(amps)
        scores=risks+epsilon*(1+amps)
        j=int(np.argmin(scores));selected.append(int(available[j]))
        trace.append(dict(size=k,objective=float(scores[j]),nominal_risk=float(risks[j]),
                          amplification=float(amps[j]),epsilon=float(epsilon)))
    return np.asarray(selected),trace
