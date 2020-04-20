import os
import re
import time
# import gzip
# import h5py
import numpy as np
import scipy.io
import scipy.sparse
import scipy.stats
import scipy.special
import pandas as pd
import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['pdf.fonttype'] = 42
plt.rcParams['ps.fonttype'] = 42
import seaborn as sns

## geom

def norm(v):
    v = np.array(v)
    normv = np.sqrt( np.sum(v**2) )
    return normv

def psi(data):
    sdata = 2* np.sqrt(data)
    return sdata

def ExpMap_s(s, w):
    s = np.array(s)
    w = np.array(w)
    r = s * np.cos(norm(w)/2) + 2. * w / norm(w) * np.sin(norm(w)/2)
    return r

def InvExpMap_s(s, r):
    s = np.array(s)
    r = np.array(r)
    costheta = 0.25 * np.sum(s * r)
    if  1.0 - costheta < 1e-9:
        w = np.zeros(s.shape)
    else:
        w = np.arccos(costheta) / np.sqrt(1. - costheta**2) * (r - s * costheta)
    return w

def RiemannianCoM_s(sdata, epsilon=1e-9, max_iter=100):
    sdata = np.array(sdata)
    (nSample, d) = sdata.shape
    t0 = time.time()
    s_k = 2. * np.ones(d, dtype=float) / np.sqrt(d)
    for k in range(max_iter):
        w = np.zeros(d, dtype=float)
        for i in range(nSample):
            si = sdata[i]
            wi = InvExpMap_s(s_k, si)
            w = w + wi / nSample
        s_k = ExpMap_s(s_k, w)
        if norm(w) < epsilon:
            # print('n_iter = ', k)
            break
    t1 = time.time()
    # print('CoM_time =', t1-t0)
    return s_k

def RotateData_s(sdata, s, r):
    sdata = np.array(sdata)
    sdata = sdata * 0.5
    s = np.array(s)
    r = np.array(r)
    (nSample, d) = sdata.shape
    t0 = time.time()
    a = s * 0.5
    b = r * 0.5
    costheta = a.dot(b)
    theta = np.arccos(costheta)
    sintheta = np.sin(theta)
    b2 = (b - a * costheta) / sintheta
    sa = sdata.dot(a)
    sb2 = sdata.dot(b2)
    sdata = sdata + np.outer( (costheta - 1) * sa - sintheta * sb2, a) + np.outer( sintheta * sa + (costheta - 1) * sb2, b2)
    sdata = sdata * 2.
    t1 = time.time()
    # print('Rotate_time =', t1-t0)
    return sdata


def count2shpere(count_data):
    count_data = np.array(count_data)
    total_count = np.sum(count_data, axis=1)
    prop_data = count_data / total_count[:,None]
    sdata = psi(prop_data)
    return sdata

def count2shpere_CoM(count_data, epsilon=1e-9, max_iter=100):
    count_data = np.array(count_data)
    total_count = np.sum(count_data, axis=1)
    prop_data = count_data / total_count[:,None]
    sdata = psi(prop_data)
    (nSample, d) = sdata.shape
    t0 = time.time()
    s_k = 2. * np.ones(d, dtype=float) / np.sqrt(d)
    for k in range(max_iter):
        w = np.zeros(d, dtype=float)
        for i in range(nSample):
            si = sdata[i]
            wi = InvExpMap_s(s_k, si)
            w = w + wi / nSample
        s_k = ExpMap_s(s_k, w)
        if norm(w) < epsilon:
            # print('n_iter = ', k)
            break
    t1 = time.time()
    # print('CoM_time =', t1-t0)
    return s_k

def count2shpere_CoM_wtd(count_data, epsilon=1e-9, max_iter=100):
    count_data = np.array(count_data)
    total_count = np.sum(count_data, axis=1)
    prop_data = count_data / total_count[:,None]
    sdata = psi(prop_data)
    wt = total_count / np.sum(total_count)
    (nSample, d) = sdata.shape
    t0 = time.time()
    s_k = 2. * np.ones(d, dtype=float) / np.sqrt(d)
    for k in range(max_iter):
        w = np.zeros(d, dtype=float)
        for i in range(nSample):
            si = sdata[i]
            wi = InvExpMap_s(s_k, si)
            w = w + wi * wt[i]
        s_k = ExpMap_s(s_k, w)
        if norm(w) < epsilon:
            # print('n_iter = ', k)
            break
    t1 = time.time()
    # print('CoM_time =', t1-t0)
    return s_k

## stat

def ml_gamma(e_lam, e_loglam, a):
    inv_a_new = 1/a + (e_loglam.mean() - np.log(e_lam.mean()) + np.log(a) - scipy.special.digamma(a))/(a - a**2*scipy.special.polygamma(1,a) )
    a_new = 1/inv_a_new
    return a_new

def ll_gamma(e_lam, e_loglam, a):
    ll = (a-1)*e_loglam.mean() - np.log(scipy.special.gamma(a)) - a*np.log(e_lam.mean()) + a*np.log(a) - a
    return ll

def m_step(e_lam, e_loglam, max_iter=10, min_iter = 3):
    lis = [0.]
    a = 0.5/( np.log(e_lam.mean()) - e_loglam.mean() )
    ll = ll_gamma(e_lam, e_loglam, a)
    lis.append(ll)
    for i in range(max_iter):
        a = ml_gamma(e_lam, e_loglam, a)
        ll = ll_gamma(e_lam, e_loglam, a)
        lis.append(ll)
        if i > min_iter and -1e-9 < lis[-1] - lis[-2] < 1e-9:
            b = a/e_lam.mean()
            return a, b, ll

def em_nb(y, m, max_iter=300, min_iter=10):
    # initialization
    m_st = scipy.stats.mstats.gmean(m)
    e_lam = np.divide(y+1/m_st, m+1)
    e_loglam = np.log(e_lam)
    lis = [0.]
    (a, b, ll) = m_step(e_lam, e_loglam)
    lis.append(ll)
    for i in range(max_iter):
        # e_step
        e_lam = np.divide( y+a , m+b )
        e_loglam = scipy.special.digamma(y+a) - np.log(m+b)
        (a, b, ll) = m_step(e_lam, e_loglam)
        lis.append(ll)
        if i > min_iter and -1e-9 < lis[-1] - lis[-2] < 1e-9:
            break
#     print( 'n_iter =', i )
    return a, b

def p_nb(y, m, a, b):
    p = scipy.stats.nbinom.pmf(y, a, b/(m+b))
    return p

def lp_nb(y, m, a, b):
    lp = scipy.stats.nbinom.logpmf(y, a, b/(m+b))
    return lp

def pval_nb(y, m, a, b):
    p_geq = 1. - scipy.stats.nbinom.cdf(y-1, a, b/(m+b))
    return p_geq

## ZINB

def f_em(y, m, pzj, max_iter=300, min_iter=10):
    # initialization
    m_st = scipy.stats.mstats.gmean(m)
    e_lam = np.divide(y+1/m_st, m)
    e_loglam = np.log(e_lam)
    fj = pzj.mean()
    k_e_lam = pzj * e_lam / fj
    k_e_loglam = pzj * e_loglam / fj
    lis = [0.]
    (a, b, ll) = m_step(k_e_lam, k_e_loglam)
    lis.append(ll)
    for i in range(max_iter):
        # e_step
        e_lam = np.divide( y+a , m+b )
        e_loglam = scipy.special.digamma(y+a) - np.log(m+b)
        k_e_lam = pzj * e_lam / fj
        k_e_loglam = pzj * e_loglam / fj
        (a, b, ll) = m_step(k_e_lam, k_e_loglam)
        lis.append(ll)
        if i > min_iter and -1e-6 < lis[-1] - lis[-2] < 1e-6:
            break
#     print( 'n_iter =', i)
    return a, b, ll


def em_zinb(y, m, max_iter=300, min_iter=10):
    # initialization
    w0 = 0.02
    (a1, b1) = em_nb(y,m)
    pmf0 = (y==0).astype('int')
    pmf1 = p_nb(y, m, a1, b1)
    ll0 = np.mean( np.log(w0*pmf0 + (1-w0)*pmf1) )
#     print(w0, a1, b1, ll0)
    lis = [ll0]
    for n in range(max_iter):
        pz0 = w0*pmf0/(w0 + (1-w0)*pmf1)
        w0 = pz0.mean()
        (a1, b1, ll) = f_em(y, m, 1.-pz0)
        pmf0 = (y==0).astype('int')
        pmf1 = p_nb(y, m, a1, b1)
        ll0 = np.mean( np.log(w0*pmf0 + (1-w0)*pmf1) )
#         print(w0, a1, b1)
        lis.append(ll0)
        if n > min_iter and (-1e-6 < lis[-1] - lis[-2] < 1e-6):
            break
    is_conv = ( n+1 < max_iter )
    return w0, a1, b1, is_conv


def pval_zinb(y, m, w0, a, b):
    p_geq = (1.-w0) * (1.- scipy.stats.nbinom.cdf(y-1, a, b/(m+b)) ) + w0 * np.array(y==0, dtype=float)
    return p_geq


def p_adjust_bh(p):
    """Benjamini-Hochberg p-value correction for multiple hypothesis testing."""
    """https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python"""
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]
