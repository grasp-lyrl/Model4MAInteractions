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
Given Qa, find Pd and simulate harm
We want to compare the cost of different Pd compared Pd*
We use the wasserstein distance between distribution

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


def simulate(w,e=100,na=int(100),debug=False):
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
    # mm = mm/mm.sum()

    return dict(mat=mat,
                harm=mm,
                Qa=w['qa'],
                Pd=Pd,
                c=c,
                C=(np.mean(c), np.std(c)))

def wasserstein_distance(qa,pd):
    ## earth movers distance in 1D, 1-norm of the CDFs
    return np.linalg.norm(np.cumsum(qa)-np.cumsum(pd), ord=1)

def hellinger_distance(qa,pd):
    ## distance between 2 discrete probability distribution
    return 1/np.sqrt(2) * np.sqrt(np.sum((np.sqrt(qa)-np.sqrt(pd))**2))

def total_variation(val1,val2):
    return 0.5*np.linalg.norm((val1 - val2), ord=1)

def similarity(x, y):
    "sum_r P1_r P2_r / sqrt[(sum_r P1_r^2) x (sum_r P2_r^2)]"
    return np.sum(x*y) / (np.sum(x**2) * np.sum(y**2))**.5

def analytical_cost(qa,pd):
    return np.sum(qa*Fbar(conv(pd)))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)

    args = parser.parse_args()

    num = 1000
    episodes = 100

    costs = np.zeros(num+1)
    stdevs = np.zeros(num+1)
    acosts = np.zeros(num+1)
    wds = np.zeros(num+1)

    ## sample Qa from lognormal
    sig = (np.log(kappa**2 + 1))**.5
    Qa = np.random.lognormal(mean=0,sigma=sig,size=N)
    Qa=Qa/Qa.sum();

    w = cross_react(Qa,fda)

    w['p_star'] = w['pd']

    s = simulate(w,e=episodes)

    costs[0] = s['C'][0]
    stdevs[0] = s['C'][1]
    acosts[0] = w['h']

    wd = wasserstein_distance(w['p_star'], w['pd'])
    wds[0] = wd


    pd_shuffle = np.copy(w['p_star'])
    for n in range(num):

        if n < num/2:
            mean = np.random.uniform(0,5) 
            std = (np.log(np.random.uniform(1,5)**2 + 1))**.5
            pd = w['p_star'] + np.random.lognormal(mean,std,N)
        else:
            mean = np.random.uniform(0,5) 
            std = (np.log(np.random.uniform(1,5)**2 + 1))**.5
            pd = w['p_star'] * np.random.lognormal(mean,std,N)

        w['pd']=pd/sum(pd);

        wd = wasserstein_distance(w['p_star'], w['pd'])

        s = simulate(w,e=episodes)


        costs[n+1] = s['C'][0]
        stdevs[n+1] = s['C'][1]
        acosts[n+1] = analytical_cost(w['qa'], w['pd'])
        wds[n+1] = wd

        if num > 10 and (n % (num//10) == 0):
            print('[%04d]'%n)


    wds, acosts, costs, stdevs = zip(*sorted(zip(wds, acosts, costs, stdevs)))


    fig, ax = plt.subplots(figsize=(8,8))

    ax.scatter(wds, acosts, label='Analytical Harm')
    ax.scatter(wds, costs, label='Empirical Harm')

    ## error bar
    for ii in range(len(wds)):
        minerror = max(costs[ii]-stdevs[ii], 0)
        ax.plot([wds[ii], wds[ii]], [costs[ii]+stdevs[ii], minerror], marker="_", color='sandybrown',alpha=0.3)

    ax.set_xlabel('$Wass_1$($P_d$, $P_d^*$)')
    ax.set_ylabel('Harm')
    plt.legend()


    if args.save:
        save_dir = './fig3/data/wass'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            
        # fig.savefig(os.path.join(save_dir,'wass.png'))
        s = dict(wds=wds, costs=costs, stdevs=stdevs, acosts=acosts)
        pickle.dump(s, open(os.path.join(save_dir, 'sim_wass.pkl'), 'wb'))

    plt.show()