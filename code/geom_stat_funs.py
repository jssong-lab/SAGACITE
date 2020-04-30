import time
# import gzip
# import h5py
import numpy as np
import scipy.stats
import scipy.special
import pandas as pd


## geom functions


def norm(v):
    v = np.array(v)
    normv = np.sqrt( np.sum(v**2) )
    return normv


def psi(data):
    sdata = 2* np.sqrt(data)
    return sdata


def psi_inv(sdata):
    data = sdata **2 / 4.
    return sdata


def ExpMap_s(s, w, epsilon=1e-9):
    s = np.array(s)
    w = np.array(w)
    if norm(w) < epsilon:
        r = s
    else:
        r = s * np.cos(norm(w)/2) + 2. * w / norm(w) * np.sin(norm(w)/2)
    return r


def InvExpMap_s(s, r, epsilon=1e-9):
    s = np.array(s)
    r = np.array(r)
    costheta = 0.25 * np.sum(s * r)
    if  1.0 - costheta < epsilon:
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


def sphere2simplex(sdata):
    sdata = np.array(sdata)
    dn = np.array(sdata < 0, dtype=int)
    sodata = (sdata - sdata * dn)**2 / 4.
    r2 = np.sum( sodata, axis=1 )
    data = sodata / r2[:, None]
    return data


## stat functions


def ml_gamma(e_lam, e_loglam, a):
    inv_a_new = 1/a + (e_loglam.mean() - np.log(e_lam.mean()) + np.log(a) - scipy.special.digamma(a))/(a - a**2*scipy.special.polygamma(1,a) )
    a_new = 1/inv_a_new
    return a_new


def ll_gamma(e_lam, e_loglam, a):
    ll = (a-1)*e_loglam.mean() - np.log(scipy.special.gamma(a)) - a*np.log(e_lam.mean()) + a*np.log(a) - a
    return ll


def m_step(e_lam, e_loglam, epsilon=1e-9, max_iter=10, min_iter = 3):
    """
    Thomas P. Minka - Estimatinga Gammadistribution
    https://tminka.github.io/papers/minka-gamma.pdf
    """
    lis = [0.]
    a = 0.5/( np.log(e_lam.mean()) - e_loglam.mean() )
    ll = ll_gamma(e_lam, e_loglam, a)
    lis.append(ll)
    for i in range(max_iter):
        a = ml_gamma(e_lam, e_loglam, a)
        ll = ll_gamma(e_lam, e_loglam, a)
        lis.append(ll)
        if i > min_iter and - epsilon < lis[-1] - lis[-2] < epsilon:
            b = a/e_lam.mean()
            return a, b, ll


def em_nb(y, m, epsilon=1e-9, max_iter=300, min_iter=10):
    """
    Thomas P. Minka - Estimatinga Gammadistribution
    https://tminka.github.io/papers/minka-gamma.pdf
    """
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
        if i > min_iter and - epsilon < lis[-1] - lis[-2] < epsilon:
            break
#     print( 'n_iter =', i )
    return a, b


def p_nb(y, m, a, b):
    p = scipy.stats.nbinom.pmf(y, a, b/(m+b))
    return p


def pval_nb(y, m, a, b):
    p_geq = 1. - scipy.stats.nbinom.cdf(y-1, a, b/(m+b))
    return p_geq


def f_em(y, m, pzj, epsilon=1e-6, max_iter=300, min_iter=10):
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
        if i > min_iter and - epsilon < lis[-1] - lis[-2] < epsilon:
            break
#     print( 'n_iter =', i)
    return a, b, ll


def em_zinb(y, m, init_w0=0.02, epsilon=1e-6, max_iter=300, min_iter=10):
    # initialization
    w0 = init_w0
    (a1, b1) = em_nb(y,m)
    pmf0 = (y==0).astype('int')
    pmf1 = p_nb(y, m, a1, b1)
    ll0 = np.mean( np.log(w0*pmf0 + (1-w0)*pmf1) )
#     print(w0, a1, b1, ll0)
    lis = [ll0]
    for n in range(max_iter):
        pz0 = w0*pmf0/(w0*pmf0 + (1-w0)*pmf1)
        w0 = pz0.mean()
        (a1, b1, ll) = f_em(y, m, 1.-pz0)
        pmf0 = (y==0).astype('int')
        pmf1 = p_nb(y, m, a1, b1)
        ll0 = np.mean( np.log(w0*pmf0 + (1-w0)*pmf1) )
#         print(w0, a1, b1)
        lis.append(ll0)
        if n > min_iter and (- epsilon < lis[-1] - lis[-2] < epsilon):
            break
    is_conv = ( n+1 < max_iter )
    return w0, a1, b1, is_conv


def pval_zinb(y, m, w0, a, b):
    y = np.asarray(y, dtype='int')
    m = np.asarray(m)
    p_geq = (1.-w0) * (1.- scipy.stats.nbinom.cdf(y-1, a, b/(m+b)) ) + w0 * np.array(y==0, dtype=float)
    return p_geq


def p_adjust_bh(p):
    """
    Benjamini-Hochberg p-value correction for multiple hypothesis testing.
    https://stackoverflow.com/questions/7450957/how-to-implement-rs-p-adjust-in-python
    """
    p = np.asfarray(p)
    by_descend = p.argsort()[::-1]
    by_orig = by_descend.argsort()
    steps = float(len(p)) / np.arange(len(p), 0, -1)
    q = np.minimum(1, np.minimum.accumulate(steps * p[by_descend]))
    return q[by_orig]


def transf_zinb(y, m, a, b, w0=0., is_log='False', log_base = np.exp(1) ):
    y = np.asarray(y, dtype='int')
    dy0 = (y==0).astype('int')
    m = np.asarray(m)
    e_lam = (1. - np.divide( w0 * dy0, ( w0 + (1-w0) * np.divide( b, m+b )**a ) ) ) * np.divide( y+a , m+b )
    if is_log=='False':
        # print('a =', a, ', b =', b, ', w =', w0, ', log:', is_log)
        return e_lam
    else:
        # print('a =', a, ', b =', b, ', w =', w0, ', log:', is_log, ', base =', log_base)
        return np.log(e_lam) / np.log(log_base)
