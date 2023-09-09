import numpy as np
import scipy as sp
from scipy.optimize import minimize, Bounds

import os, sys, pdb, random, datetime, pickle, argparse
from ipdb import set_trace as st

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt

plt.rcParams['figure.figsize'] = [8,8]
sns.set_theme()
fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=0.8*fsz)
plt.rc('xtick', labelsize=0.7*fsz)
plt.rc('ytick', labelsize=0.7*fsz)
plt.rc('legend', fontsize=0.8*fsz)
plt.rc('figure', titlesize=fsz)
# plt.rc('pdf', fonttype=42)
sns.set_style("ticks", rc={"axes.grid":True})


"""
Written by Christopher Hsu

Simulate different base case with different number of agents to
test limits of the theory

use click args to choose case when Na=Nd or Na!=Nd
"""

def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
setup(42)

# gaussian fra bandwidth
sigma=0.05
D = 0.1*sigma
beta = 1/110.

# discretization of the axis
L = 1.
N=int(L/D);
N = N if N % 2 == 0 else N+1;

# sigma/mu for log-normal that is creates the distribution of antigens qa
kappa=5;

# eqn C1-C2 which define growth of antigen a
nu_a = 1.; nup_a = 1.;
alpha=nu_a/nup_a;

## discretize state
x = np.linspace(0,1,N,endpoint=False)

## functions
Fbar = lambda pa: sp.special.gamma(1+alpha)/pa**alpha
dFbar = lambda pa: -alpha*sp.special.gamma(1+alpha)/pa**(1+alpha)
Fm = lambda m: m**alpha

def make_convolve_1d_pbc(B=1.0):
    ## Modified from https://github.com/andim/optimmune
    frp = np.zeros_like(x)
    for shift in np.arange(-5.0, 5.1, 1):
        frp += np.exp(-(x+shift)**2/2/sigma**2)
    frp_ft = np.fft.rfft(frp)

    def convolve(f):
        return np.fft.irfft(np.fft.rfft(f) * frp_ft)
    return convolve

conv = make_convolve_1d_pbc(x)

def func(x, sigma):
    ## Gaussian
    return np.exp(- x**2 / (2.0 * sigma**2))

def build_1d_fda_matrix(func, x, sigma, B=1):
    """ Builds quadratic fda matrix respecting pbc.

    func: Kernel function
    x: position of points
    sigma: width of Kernel

    Modified from https://github.com/andim/optimmune
    """
    N = len(x)
    A = np.zeros((N, N))
    shifts = np.arange(-5, 6) * B
    for r in range(N):
        for p in range(N):
            value = 0
            for shift in shifts:
                value += func(x[r] - x[p] + shift, sigma[r])
            A[r, p] = value
    return A

sigmas = np.ones(N)*sigma
fda = build_1d_fda_matrix(func, x, sigmas)

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
    return dict(qa=qa,pd=pd,fda=fda,x=x,N=N,h=h)

def simulate(w,e=100,na=int(10),nd=int(10),debug=False):
    """
    fat is the number of attackers at time t
    lat is the rate of interaction of a, each interaction at time t happens to a defenders
    d drawn from Pd.
    fda defines the probability of Bernoulli RV of recognition of a from d
    upon such successful recognition, the harm caused by a is simply mat (the number of
    attackers of that particular a at the time instant of interaction). 
    after the response we set that particular fat and lat to zero.
    """

    N = w['N']
    c = []
    mat_list = []

    T=N
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

            ## for nd != na, adjust idx of attackers interactions
            ## if num attackers sampled > number of def
            if a.sum() > nd:
                idx = np.random.choice(idx, nd, replace=False)
                idx.sort()
                d = np.random.multinomial(nd, Pd)
            ## else sample from attacker interacted with from Pd
            ## extra defenders does not help
            else:
                d = np.random.multinomial(a.sum(), Pd)
            d = np.repeat(np.arange(N), d)
            np.random.shuffle(d)

            p=w['fda'][d, qa[idx]].reshape(-1)

            recog = (np.random.random(len(p)) < p).astype(int)
            response = np.zeros(len(qa))
            response[idx[recog==1]] = 1

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
    # mm = mm/mm.sum()

    return dict(mat=mat,
                harm=mm,
                Qa=w['qa'],
                Pd=Pd,
                c=c,
                C=(np.mean(c), np.std(c)))


def differing(save):
    ## sample Qa from lognormal
    sig = (np.log(kappa**2 + 1))**.5
    Qa = np.random.lognormal(mean=0,sigma=sig,size=N)
    Qa=Qa/sum(Qa);

    ## get optimal Pd
    w = cross_react(Qa, fda)

    max_agents = 101
    every = 5

    Nas = []
    Nds = []
    costs = []
    stdev = []


    for j in range(1, max_agents, every):
        for k in range(1, max_agents, every):
            Nas.append(j)
            Nds.append(k)

            s = simulate(w,e=100,na=int(j), nd=int(k), debug=False)

            costs.append(s['C'][0])
            stdev.append(s['C'][1])

        if max_agents > 10 and (j % (max_agents-1//10) == 0):
            print('[%04d]'%j)

    Nas = np.asarray(Nas)
    Nds = np.asarray(Nds)
    costs = np.asarray(costs)
    stdev = np.asarray(stdev)


    fig1 = plt.figure()
    ax = fig1.add_subplot(projection='3d')


    ax.scatter(Nas,Nds,costs, c=costs, cmap='viridis')

    for ii in range(len(Nas)):
        minerror = max(costs[ii]-stdev[ii], 0)
        ax.plot([Nas[ii], Nas[ii]], [Nds[ii], Nds[ii]], [costs[ii]+stdev[ii], minerror], marker="_", color='k',alpha=0.3)

    ax.view_init(20,35)

    ax.set_xlabel("\n\n # of Attackers")
    ax.set_ylabel("\n\n # of Defenders")
    ax.set_zlabel("Harm",rotation=90)

    plt.tight_layout()

    if save:
        w = dict(Nas=Nas,
                 Nds=Nds,
                 costs=costs,
                 stdev=stdev
                )

        save_dir = './fig4/data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fig1.savefig(os.path.join(save_dir,'limits_diff.png'))
        pickle.dump(w, open(os.path.join(save_dir,'w_diff.pkl'), 'wb'))

    plt.show()

def same(save):
    ## sample Qa from lognormal
    sig = (np.log(kappa**2 + 1))**.5
    Qa = np.random.lognormal(mean=0,sigma=sig,size=N)
    Qa=Qa/sum(Qa);

    ## get optimal Pd
    w = cross_react(Qa, fda)

    max_agents = 1001
    every = 5

    Nas = []
    costs = []
    stdev = []


    for j in range(1, max_agents, every):
        Nas.append(j)

        s = simulate(w,e=100,na=int(j), nd=int(j), debug=False)

        costs.append(s['C'][0])
        stdev.append(s['C'][1])

        if max_agents > 10 and (j % ((max_agents-1)//10) == 0):
            print('[%04d]'%j)

    Nas = np.asarray(Nas)
    costs = np.asarray(costs)
    stdev = np.asarray(stdev)


    fig, ax = plt.subplots(figsize=(8,8))

    plt.plot(np.arange(max_agents), np.ones(max_agents)*w['h'], label='Analytical Harm')
    ax.scatter(Nas,costs, color='sandybrown', label='Empirical Harm')


    for ii in range(len(Nas)):
        minerror = max(costs[ii]-stdev[ii], 0)
        ax.plot([Nas[ii], Nas[ii]], [costs[ii]+stdev[ii], minerror], marker="_", color='k',alpha=0.3)

    ax.set_xlabel("# of agents per team")
    ax.set_ylabel("Harm")

    plt.tight_layout()

    if save:
        w = dict(Nas=Nas,
                 costs=costs,
                 stdev=stdev,
                 harm=w['h']
                )

        save_dir = './fig4/data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fig.savefig(os.path.join(save_dir,'limits_same.png'))
        pickle.dump(w, open(os.path.join(save_dir,'w_same.pkl'), 'wb'))

    plt.show()

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--same', type=int, default=1)   ## same number of agents exp or differing
    args = parser.parse_args()

    if args.same:
        same(args.save)
    else:
        differing(args.save)

