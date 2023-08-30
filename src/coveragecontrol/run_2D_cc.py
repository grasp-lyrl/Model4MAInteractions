import torch as th, torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy as sp
from scipy.optimize import minimize, Bounds, LinearConstraint

import os, sys, pdb, random, json, gzip, bz2, pdb, datetime, pickle, argparse
from ipdb import set_trace as st
from copy import deepcopy
from collections import defaultdict
from functools import partial

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.gridspec as gridspec
# %matplotlib inline
plt.rcParams['figure.figsize'] = [8,8]

"""
Written by Christopher Hsu

Given Qa, find Pd and simulate harm in the 2D case
"""

def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
    prng = np.random.RandomState(seed)
    return prng
prng = setup(42)

# gaussian fra bandwidth
sigma=0.05
# D = 0.5*sigma
D = 0.05
beta = 1/110.
xi = 10*sigma

# discretization of the axis
L = 1.
N=int(L/D);
N = N if N % 2 == 0 else N+1;

# sigma/mu for log-normal that is creates the distribution of antigens qa
kappa=5;

# eqn C1-C2 which define growth of antigen a
nu_a = 1; nup_a = 1.;
alpha=nu_a/nup_a;

## discretize state
x = np.linspace(0,1,N,endpoint=False)

## functions
Fbar = lambda pa: sp.special.gamma(1+alpha)/pa**alpha
dFbar = lambda pa: -alpha*sp.special.gamma(1+alpha)/pa**(1+alpha)
Fm = lambda m: m**alpha

def correlated_random_cutoff_2d(cv, N, q0, alpha=2.0, prng=np.random):
    # generate normal random variables
    sig = (np.log(cv**2 + 1))**.5
    ux = prng.normal(0.0, sig, (N, N))
    uq = np.fft.fft2(ux)
    q = np.abs(np.fft.fftfreq(N, d=1./N)[:uq.shape[0]])
    q = 2 * np.pi * q
    q1, q2 = np.meshgrid(q, q)
    S = 1.0/(1.0+((q1**2 + q2**2)**.5/q0)**alpha)
    S /= np.mean(S)
    etaq = S**.5 * uq
    etax = np.real(np.fft.ifft2(etaq))
    lognorm = np.exp(etax)
    lognorm /= np.sum(lognorm)
    return lognorm

def func(x, sigma):
    ## Gaussian
    return np.exp(- x**2 / (2.0 * sigma**2))

def make_convolve_2d_pbc(func, x, sigma, B=1.0):
    N = len(x)
    # indexing = ij only available in new numpy versions
    # xv, yv = np.meshgrid(x, x, indexing='ij')
    # this is equivalent
    yv, xv = np.meshgrid(x, x)
    frp = np.zeros_like(xv)
    for xshift in np.arange(-5.0, 5.1, B):
        for yshift in np.arange(-5.0, 5.1, B):
            frp += func(((xv+xshift)**2 + (yv+yshift)**2)**.5, sigma)

    frp_ft = np.fft.rfft2(frp)
    fft = np.fft.rfft2
    ifft = np.fft.irfft2
    def convolve(x):
        return ifft(fft(x.reshape((N, N))) * frp_ft).flatten()
    return convolve

def build_2d_frp_matrix(func, vector, sigma):
    """ Builds quadratic frp matrix respecting pbc.

    func: Kernel function
    vector: vector of equally spaced points
    """
    spacing = np.diff(vector)[0]
    Nvec = len(vector)
    nn = Nvec**2
    A = np.zeros((nn, nn))
    func0 = func(0, sigma)

    for row in range(nn):
        x1 = row % Nvec
        y1 = row / Nvec
        A[row, row] = func0
        for col in range(row):
            x2 = col % Nvec
            y2 = col / Nvec
            value = 0
            for xshift in range(-1,2):
                for yshift in range(-1,2):
                    value += func( spacing * ((x1-x2+xshift*Nvec)**2 + (y1-y2+yshift*Nvec)**2)**.5 , sigma)
            A[row, col] = value
            A[col, row] = value
        if nn > 10 and (row % (nn//10) == 0):
            print('[%04d]'%row)
    return A

def calc_pd(qa):
    N = len(qa)

    def obj(pd):
        return np.sum(qa*Fbar(conv(pd)))
    def grad(pd):
        return conv(qa*dFbar(conv(pd)))
    def cons(pd):
        return pd.sum()-1

    pd0 = np.ones(N)/float(N)
    ## optimize with constraints
    r = minimize(lambda pd: obj(pd), pd0,method='SLSQP',
                 jac=lambda pd: grad(pd),
                 bounds=Bounds(0,np.inf),
                 constraints=[{'type': 'eq', 'fun':cons}],
                 tol=1e-8)
    pd = r['x']/r['x'].sum()
    return dict(pd=pd,h=obj(pd))

def cross_react(qa,fda):
    r = calc_pd(qa);
    pd=r['pd']
    h =r['h']
    # print('h*: %2.3f'%h)
    return dict(qa=qa,pd=pd,fda=fda,x=x,N=len(qa),h=h)

def simulate(w,e=100,na=int(1000),debug=False):
    """
    fat is the number of attackers at time t
    lat is the rate of interaction of a, each interaction at time t happens to a defenders
    d drawn from Pd.
    fda defines the probability of Bernoulli RV of recognition of a from d
    upon such successful recognition, the harm caused by a is simply mat (the number of
    attackers of that particular a at the time instant of interaction). 
    after the response we set that particular fat and lat to zero.
    """
    mat_list = []
    N = w['N']
    c = []
    T=N*10
    for ee in range(e):
        ## sample na attackers from Qa
        qa = np.random.multinomial(na, w['qa'])
        qa = np.repeat(np.arange(N), qa)
        ## get bin indexs for agents
        qas = np.tile(qa, (N,1))
        oo = np.arange(N)
        oo = np.tile(oo, (na,1)).T
        ii = (qas==oo)

        if debug:
            print('qa', qa)
            print('pd', w['pd'])
        fat = np.ones(len(qa)); lat = np.ones(len(qa)); mat = np.zeros(len(qa))

        for t in range(T):

            """
            qa is a list of all attackers present in the environment,
            a is dot{m}at the rate of the mat, i.e. attackers that have an interaction
            so we can use qa[idx] to select the indices of the attackers for which interaction happened
            d are the defenders sampled that have randomly interacted with an attacker
            """

            Pd = w['pd']

            a = (np.random.poisson(lat) > 0).astype(int);
            mat += a
            idx = np.nonzero(a)[0]

            d = np.random.multinomial(a.sum(), Pd)
            d = np.repeat(np.arange(N), d)
            np.random.shuffle(d)

            ## probability of interaction
            p=w['fda'][d, qa[idx]].reshape(-1)

            ## bernoulli rv of interaction (1=recognition, 0=unsuccessful)
            recog = (np.random.random(len(p)) < p).astype(int)
            response = np.zeros(len(qa))
            response[idx[recog==1]] = 1

            ## if response is successful
            fat[response > 0] = 0
            lat[response > 0] = 0

            if (t % 1 == 0) and debug:
                print('[%04d]'%t)
                print('a', a)
                print('d', d)
                print('p', p)
                print('rec', recog)
                print('res', response)
                print('lat', lat)

            ## exponential increase of rate
            fat += (fat >0)*nu_a
            lat += (lat >0)*nup_a

            ## sim ending condition
            if ((fat>0).sum() == 0) or t == T-1:
                if debug:
                    print('[%05d]'%t)
                    print('fat', fat)
                ## harm calculation for episode
                c.append( Fm(mat).mean())
                break

        ## episode harm distribution from na attackers to N bins
        xx = np.tile(Fm(mat), (N,1))
        mm = np.sum(xx*ii, axis=1)
        mm = mm/N
        mm = mm/mm.sum()
        mat_list.append(mm)
    ## average over all episodes harm and smooth with conv
    mm = np.asarray(mat_list)
    mm = np.sum(mm,axis=0)
    mm = conv(mm)/conv(mm).sum()
    return dict(mat=mat,
                harm=mm,
                Qa=w['qa'],
                Pd=Pd,
                c=c,
                C=(np.mean(c), np.std(c)))


if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--sigma_P', type=float, default=0.08)

    args = parser.parse_args()


    sigma_Q = 0.05
    conv = make_convolve_2d_pbc(func, x, sigma_Q)

    Qa = correlated_random_cutoff_2d(kappa, N, 1.0 / xi, alpha=2.0,
                                           prng=prng).flatten()
    Qa = conv(Qa)/conv(Qa).sum()


    sigma_P = args.sigma_P
    conv = make_convolve_2d_pbc(func, x, sigma_P)
    fda = build_2d_frp_matrix(func, x, sigma_P)

    w = cross_react(Qa,fda)

    s = simulate(w,e=100)

    print('Cost: %2.3f +/- %2.3f, Analytical: %2.3f'%
                                (s['C'][0], s['C'][1], w['h']))

    fig = plt.figure(figsize=(15,8))
    grid = gridspec.GridSpec(1, 3)
    ax = plt.Subplot(fig, grid[0, 0])
    fig.add_subplot(ax)    
    ax.imshow(s['Qa'].reshape(N,N),
                label='$Q_a$')
    plt.title('$Q_a$')

    ax1 = plt.Subplot(fig, grid[0, 1])
    fig.add_subplot(ax1)    
    ax1.imshow(w['pd'].reshape(N,N),
                label='$P_r^*$')
    plt.title('$P_r^*$')
    

    m = s['harm']
    # m = conv(m)/conv(m).sum()
    # m = conv(m/m.sum());
    ax2 = plt.Subplot(fig, grid[0, 2])
    fig.add_subplot(ax2)  
    ax2.imshow(m.reshape(N,N),
                label='Harm');
    plt.title('Harm')
    plt.tight_layout()


    if args.save:
        save_dir = './coveragecontrol/data/base%d'%(sigma_P*100)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        fig.savefig(os.path.join(save_dir,'base.png'))
        pickle.dump(s, open(os.path.join(save_dir,'sim.pkl'), 'wb'))
        pickle.dump(w, open(os.path.join(save_dir,'w_react.pkl'), 'wb'))
    
    plt.show()
    # pdb.set_trace()