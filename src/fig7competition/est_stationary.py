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

Given Qa, find Pd with competition and simulate harm
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
phi = lambda pa: alpha*sp.special.gamma(1+alpha)/pa**(1+alpha)

# death rate
ded = 0.001

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
    wass = []
    qhats = []
    Nds = []

    ## reset population inits
    Nd = np.ones(N)/N
    Nd_tot = na
    Nd *= Nd_tot

    ## reset KF stats
    mu_belief_m = np.ones(na)/na
    cov_belief = np.eye(na)*0.5
    ## reset qhat beliefs
    qhat = np.ones(N)/N
    mu_m = qhat


    T=100
    for ee in range(e):
        total = 0
        ## sample na attackers from Qa
        qa1 = np.random.multinomial(na, w['qa'])
        qa = np.repeat(np.arange(N), qa1)
        ## get bin indexs for agents
        qas = np.tile(qa, (N,1))
        oo = np.arange(N)
        oo = np.tile(oo, (na,1)).T
        ii = (qas==oo)

        if debug:
            print('qa', qa)
            print('pd', w['pd'])
        fat = np.ones(len(qa)); lat = np.ones(len(qa)); mat = np.zeros(len(qa))

        dt = 1e4/50

        for t in range(T):

            """
            qa is a list of all attackers present in the environment,
            a is dot{m}at the rate of the mat, i.e. attackers that have an interaction
            so we can use qa[idx] to select the indices of the attackers for which interaction happened
            d are the defenders sampled that have randomly interacted with an attacker
            """

            ## competition dynamics
            Nd += dt * Nd * (conv(qhat * phi(conv(Nd))) - ded)
            Pd = Nd/Nd.sum()

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

            if not args.KF:
                xx = np.tile(mat, (N,1))
                qhat = np.sum(xx*ii, axis=1)
                qhat = qhat/qhat.sum()

            else:

                """Kalman Filter: Belief update of Qa given observations
                KF is an estimate of all na antigens
                """

                ## C is an interaction of antigen, dot{m}at, a bool
                C = np.copy(a)

                var = 0.5
                z = np.random.normal(0,var)
                ## observation noise
                obs_noise = np.ones(na)* 2 * z
                obs_noise[C==1] = z  
                observation = C + obs_noise
                observation[response==1] = mat[response==1] +z
                ## observation variance noise
                Q = np.ones(na) * 4 * var**2
                Q[C==1] = var**2

                ## KF observation update
                K = cov_belief @ C.T / (C @ cov_belief @ C.T + Q)
                mu_belief = mu_belief_m + K*(observation - C*mu_belief_m)
                cov_belief = (np.eye(na) - np.outer(K,C)) @ cov_belief

                """We now have distributed observations of each type
                We want combine these multiple observations to create a single estimate
                of each type (size N) to produce a centralized KF estimate of qhat
                """
                ww = np.tile(mu_belief, (N,1))
                # mean of Nxna matrix -> N
                ss = np.sum(ww * ii, axis=1)
                mu = np.divide(ss, qa1, out=np.zeros_like(ss), where=qa1!=0)
                
                """ reocover Qa_hat from state
                original: Ma(t) = Ma(t-1)e^(nu)
                nu = log(ma(t)/ma(t-1))
                qhat = Ma(0) = Ma(t)e^(-nu*t)
                """
                nu = mu_m / (mu+1e-6)
                mao = mu*nu**t

                ## in case ma(0) is negative
                mao = np.maximum(mao, np.zeros(N))
                qhat = mao/mao.sum()

                ## update prev mu with mu
                mu_m = mu
                mu_belief_m = mu_belief

            ## sim ending condition
            if ((fat>0).sum() == 0) or t == T-1:
                if debug:
                    print('[%05d]'%t)
                    print('fat', fat)
                ## harm calculation for episode
                c.append( Fm(mat).mean())
                ## calc distances
                wass.append(wasserstein_distance(w['pd'],Pd))
                Nds.append(Nd)
                qhats.append(qhat)
                break

        if e > 10 and (ee % (e//10) == 0):
            print('[%04d]'%ee)

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
                qhats=qhats,
                Nds=Nds,
                Pd=Pd,
                wass=np.asarray(wass),
                c=c,
                C=(np.mean(c), np.std(c)))

def wasserstein_distance(qa,pd):
    ## earth movers distance in 1D, 1-norm of the CDFs
    return np.linalg.norm(np.cumsum(qa)-np.cumsum(pd), ord=1)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--KF', type=int, default=0)

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
    plt.plot(w['x'],w['pd'],label='opt $P_d^*$');
    plt.plot(w['x'],s['Pd'],label='comp $P_d^*$');
    plt.plot(w['x'],s['harm'],label='Harm');
    plt.legend();
    plt.xlabel('State (x)')
    plt.ylabel('Probability')
    plt.tight_layout()

    if args.save:
        save_dir = './fig7competition/data/est_state'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fig.savefig(os.path.join(save_dir,'comp_est.png'))
        pickle.dump(s, open(os.path.join(save_dir,'sim.pkl'), 'wb'))
        pickle.dump(w, open(os.path.join(save_dir,'w_react.pkl'), 'wb'))
    
    plt.show()