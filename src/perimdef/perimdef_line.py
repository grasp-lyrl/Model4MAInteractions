import numpy as np
import scipy as sp
from scipy.optimize import minimize, Bounds

import os, sys, pdb, random, datetime, pickle, argparse
from ipdb import set_trace as st

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
plt.rcParams['figure.figsize'] = [8,8]


"""
Written by Christopher Hsu

Modified from https://github.com/andim/optimmune

Given Qa, find Pd and simulate harm
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

def limit_vel(control, max_v):
    return np.clip(control, -max_v, max_v)

def simulate(w,e=100,na=int(100),debug=False,render=True):
    if render:
        fig = plt.figure()
        fig.clf()

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

        # for t in range(T):

        """
        qa is a list of all attackers present in the environment,
        a is dot{m}at the rate of the mat, i.e. attackers that have an interaction
        so we can use qa[idx] to select the indices of the attackers for which interaction happened
        d are the defenders sampled that have randomly interacted with an attacker
        """

        Pd = w['pd']

        ## interactions happen at a poisson rate lat. a are the attacker sthat have interacted
        # a = (np.random.poisson(lat) > 0).astype(int);
        # mat += a
        # idx = np.nonzero(a)[0]
        ## num of attackers that have interacted
        # nn = a.sum()

        ## interacting defenders
        d = np.random.multinomial(na, Pd)
        d = np.repeat(np.arange(N), d)
        ## random assigment
        np.random.shuffle(d)

        ## initialize intruders and defenders in cartesian
        ## we want to also map to arbitrary convex shape based on arc length
        intruders = np.zeros((na,2))
        defenders = np.zeros((na,2))

        for jj, aa in enumerate(qa):
        # for jj, aa in enumerate(qa[idx]):
            a_loc = w['x'][aa]
            intruders[jj,:] = [a_loc,1]

        for jj, dd in enumerate(d):
            d_loc = w['x'][dd]
            defenders[jj,:] = [d_loc,0]

        ## figure out ratio between cross rxn and attacker to reach to perim
        ## so if att gets there in 10 steps max cross rxn should be 10
        ## 10/200 = 0.05
        ## att velocity in y
        v_a = -0.1
        ## def max velocity in x
        v_d = 0.05

        T = 10
        for tt in range(T):
            ## def already assigned an intruder. will need to adjust for convex shape
            control = intruders[:,0] - defenders[:,0]
            ## update positions
            intruders[:,1] += v_a
            defenders[:,0] += limit_vel(control, max_v=v_d)

            ## probability of interaction
            # p=w['fda'][d, qa[idx]].reshape(-1)

            ## bernoulli rv of interaction (1=recognition, 0=unsuccessful)
            # recog = (np.random.random(len(p)) < p).astype(int)
            # response = np.zeros(len(qa))
            # response[idx[recog==1]] = 1

            ## response if def is within eps dist from att
            eps = np.ones(na)*D
            dist = np.linalg.norm(intruders - defenders, axis=1) 
            recog = (dist <= eps).astype(int)
            response = np.zeros(len(qa))
            # response[idx[recog==1]] = 1
            response[recog==1] = 1

            ## if response is successful
            fat[response > 0] = 0
            # lat[response > 0] = 0

            ## exponential increase of rate
            fat += (fat >0)*nu_a
            # lat += (lat >0)*nup_a

            if render:
                fig.clf()
                ax = fig.add_subplot()
                ax.plot(w['x'], w['qa'],label='$Q_a$', linewidth=2.);
                ax.plot(w['x'], w['pd'],label='$P_{d,u}^*$', linewidth=2.);

                ax.set_xlabel('State (x)')
                ax.set_ylabel('Probability')
                plt.ylim(-0.002,0.20)
                plt.legend()

                sns.despine(ax=ax)

                sns.set_style("ticks", rc={"axes.grid":True})
                ax1 = ax.twinx()

                color = 'tab:blue'
                ax1.scatter(intruders[:,0], intruders[:,1], marker='.',label='intruder');
                ax1.scatter(defenders[:,0], defenders[:,1], marker='.',label='defender');

                ax1.set_ylabel('State (y)', color=color)
                ax1.set_ylim(-0.01,1.0)
                ax1.tick_params(axis='y', labelcolor=color)
                
                plt.tight_layout()
                sns.despine(ax=ax1, right=False)

                plt.draw()
                plt.pause(0.001)

            ## sim ending condition
            # if ((fat>0).sum() == 0) or t == T-1:
                # if debug:
                #     print('[%05d]'%t)
                #     print('fat', fat)
                ## harm calculation for episode
                # c.append( Fm(mat).mean())
        c.append(recog.sum()/na)
        print(recog.sum()/na)
                # break

        if e > 10 and (ee % (e//10) == 0):
            print('[%04d]'%ee)

        ## episode harm distribution from na attackers to N bins
        # xx = np.tile(Fm(mat), (N,1))
        # mm = np.sum(xx*ii, axis=1)
        # mm = mm/N
        # mm = mm/mm.sum()
        # mat_list.append(mm)
    ## average over all episodes harm and smooth with conv
    # mm = np.asarray(mat_list)
    # mm = np.sum(mm,axis=0)
    # mm = conv(mm)/conv(mm).sum()
    # mm = mm/mm.sum()

    return dict(mat=mat,
                harm=mm,
                Qa=w['qa'],
                Pd=Pd,
                c=c,
                C=(np.mean(c), np.std(c)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)

    args = parser.parse_args()

    ## sample Qa from lognormal
    sig = (np.log(kappa**2 + 1))**.5
    Qa = np.random.lognormal(mean=0,sigma=sig,size=N)
    Qa=Qa/Qa.sum();

    w = cross_react(Qa,fda)
    s = simulate(w,e=100, debug=False)

    print('Empirical Harm: %2.3f +/- %2.3f, Analytical: %2.3f'%
                                (s['C'][0], s['C'][1], w['h']))

    fig = plt.figure(); plt.clf();
    plt.plot(w['x'],w['qa'],label='$Q_a$');
    plt.plot(w['x'],w['pd'],label='$P_d^*$');
    plt.plot(w['x'],s['harm'],label='Harm');
    plt.legend();
    plt.xlabel('State (x)')
    plt.ylabel('Probability')
    plt.tight_layout()


    if args.save:
        save_dir = './pd/data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fig.savefig(os.path.join(save_dir,'pd.png'))
        pickle.dump(s, open(os.path.join(save_dir,'sim.pkl'), 'wb'))
        pickle.dump(w, open(os.path.join(save_dir,'w_react.pkl'), 'wb'))
    
    plt.show()