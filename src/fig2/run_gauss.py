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

Figure 2, Section IV: optimal defender distirbution is supported on a discrete set

Given Gaussian Qa, find Pd given a nonuniform bandwidth sigma and simulate harm
"""

def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
setup(42)

# gaussian fra bandwidth
sigma=0.05
D = 0.02
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

def calc_pd(qa):
    N = len(qa)

    def obj(pd):
        Ptilde = np.dot(pd, fda)
        f = np.sum(qa*Fbar(Ptilde))
        return f
    def grad(pd):
        Ptilde = np.dot(pd, fda)
        return np.dot(fda, qa*dFbar(Ptilde))
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


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)


    args = parser.parse_args()

    mu = 0.5
    sig_Q = 0.1
    print('sig_Q', sig_Q, 'threshold',sig_Q*np.sqrt(2))
    a = np.linspace(0,1,N)
    qa = np.exp(-(a-mu)**2/2./sig_Q**2)
    Qa = qa/qa.sum()

    sigs = []
    acosts = []
    costs = []
    stdev = []
    pds = []
    diversity = []

    every = 5
    top = int(0.2/0.001)

    for j in range(1,top,every):
        sigs.append(0.001*j)
        
        sigmas = np.ones(N)*0.001*j
        fda = build_1d_fda_matrix(func, x, sigmas)

        w = cross_react(Qa,fda)
        s = simulate(w,e=100,na=int(100), debug=False)


        w['pd'][w['pd'] < 1e-8] = 0.0
        diversity.append(np.linalg.norm(w['pd'], ord=0))
        pds.append(w['pd'])
        acosts.append(w['h'])
        costs.append(s['C'][0])
        stdev.append(s['C'][1])

        if j > 10 and (j % ((top-1)//10) == 0):
            print('[%04d]'%j)

    sigs = np.asarray(sigs)
    acosts = np.asarray(acosts)
    costs = np.asarray(costs)
    stdev = np.asarray(stdev)
    diversity = np.asarray(diversity)

    print('Empirical Harm: %2.3f +/- %2.3f, Analytical: %2.3f'%
                                (s['C'][0], s['C'][1], w['h']))

    fig = plt.figure(); plt.clf();
    plt.plot(w['x'],w['qa'],label='$Q_a$');
    plt.plot(w['x'],w['pd'],label='$P_d^*$');
    plt.plot(w['x'],pds[-1],label='$P_d^*$');
    plt.plot(w['x'],s['harm'],label='Harm');
    plt.legend();
    plt.xlabel('State (x)')
    plt.ylabel('Probability')
    plt.tight_layout()


    fig2, ax2 = plt.subplots()
    color = 'tab:red'
    ax2.set_xlabel('Cross Reactivity Bandwidth $\sigma$')
    plt.plot(sigs, acosts, label='Analytical Harm')
    ax2.scatter(sigs, costs, color=color, label='Empirical Harm')

    for ii in range(len(sigs)):
        minerror = max(costs[ii]-stdev[ii], 0)
        ax2.plot([sigs[ii], sigs[ii]], [costs[ii]+stdev[ii], minerror], marker="_", color='k',alpha=0.3)
    ax2.set_ylabel('harm', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    ax3 = ax2.twinx()
    color = 'tab:blue'
    ax3.plot(sigs, diversity, label='', color=color)
    ax3.set_ylabel('Diversity of Defense ($||P_d^*||_0$)')
    ax3.tick_params(axis='y', labelcolor=color)
    plt.legend()
    fig2.tight_layout()


    if args.save:
        save_dir = './fig2/data'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fig.savefig(os.path.join(save_dir,'gauss_probs.pg'))
        # fig2.savefig(os.path.join(save_dir,'gauss_harms.png'))

        s = dict(sigs=sigs,
             costs=costs,
             acosts=acosts,
             stdev=stdev,
             h=w['h'],
             diversity=diversity,
             pds=pds
            )

        pickle.dump(s, open(os.path.join(save_dir,'sim_sigs.pkl'), 'wb'))
        pickle.dump(w, open(os.path.join(save_dir,'w_react_sigs.pkl'), 'wb'))
    
    plt.show()