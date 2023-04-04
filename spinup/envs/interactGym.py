import torch as th, torch.nn as nn
import numpy as np
import torch.nn.functional as F
import scipy as sp
from scipy.optimize import minimize, Bounds, LinearConstraint

import os, sys, pdb, random, json, gzip, bz2, datetime, pickle, argparse
from ipdb import set_trace as st
from copy import deepcopy
from collections import defaultdict
from functools import partial

import gym
from gym import spaces

import matplotlib.pyplot as plt
# %matplotlib inline
plt.rcParams['figure.figsize'] = [8,8]

import seaborn as sns
sns.set_context('talk')
sns.set(context='notebook',
        style='ticks',
        font_scale=1,
        rc={'axes.grid':True,
            'grid.color':'.9',
            'grid.linewidth':0.75})        
""" 
Written by Christopher Hsu

"""
def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
setup(88)

class InteractionProblem(object):
    def __init__(self,na,nd):
        
        ## number of attackers and defenders
        self.na=int(na)
        self.nd=int(nd)
        ## gaussian fra bandwidth
        self.sigma=0.05
        self.D = 0.1*self.sigma
        self.beta = 1/110.
        self.xi = 5*self.sigma
        ## discretization of the axis
        self.L = 1.
        self.N=int(self.L/self.D);
        self.N = self.N if self.N % 2 == 0 else self.N+1;
        ## sigma/mu for log-normal that is creates the distribution of antigens qa
        self.kappa=5;
        ## eqn C1-C2 which define growth of antigen a
        self.nu_a = 1
        self.nup_a = 1.;
        self.alpha=self.nu_a/self.nup_a;
        ## state space
        self.x = np.linspace(0,1,self.N,endpoint=False)
        # self.gaussian = np.exp(-self.x**2/2./self.sigma**2)
        ## harm functions
        self.Fbar = lambda pa: sp.special.gamma(1+self.alpha)/pa**self.alpha
        self.dFbar = lambda pa: -self.alpha*sp.special.gamma(1+self.alpha)/pa**(1+self.alpha)
        self.Fm = lambda m: m**self.alpha
        self.A = lambda pa: self.alpha*sp.special.gamma(1+self.alpha)/pa**(1+self.alpha)
        ## cross reactivity
        self.conv = self.make_convolve_1d_pbc(self.x)
        self.sigmas = np.ones(self.N)*self.sigma
        self.fda = self.build_1d_fda_matrix(self.func,self.x,self.sigmas)
        ## RL definitions
        self.observation_space = spaces.Box(np.float32(np.zeros(self.N)),np.float32(np.ones(self.N)))
        self.action_space = spaces.Box(np.float32(-np.ones(self.nd)), np.float32(np.ones(self.nd)))


        ## init random Qa
        s = (np.log(self.kappa**2 + 1))**.5
        Qa = np.random.lognormal(mean=0,sigma=s,size=self.N)
        self.Qa=Qa/sum(Qa)

        # y = np.linspace(0.0, 2*np.pi, self.N, endpoint=False)
        # Qa = np.sin(y*(6)) + 1
        # self.Qa=Qa/np.sum(Qa)

        self.Pd = np.ones(self.N)/self.N
        self.Pdstar = self.Pd

    def make_convolve_1d_pbc(self, B=1.0):
        frp = np.zeros_like(self.x)
        for shift in np.arange(-5.0, 5.1, 1):
            frp += np.exp(-(self.x+shift)**2/2/self.sigma**2)
        frp_ft = np.fft.rfft(frp)

        def convolve(f):
            return np.fft.irfft(np.fft.rfft(f) * frp_ft)
        return convolve

    def func(self, x, sigma):
        ## Gaussian
        return np.exp(- x**2 / (2.0 * sigma**2))

    def build_1d_fda_matrix(self, func, x, sigma, B=1):
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

    def calc_pd(self,qa):

        def obj(pd):
            return np.sum(qa*self.Fbar(self.conv(pd)))
        def grad(pd):
            return self.conv(qa*self.dFbar(self.conv(pd)))
        def cons(pd):
            return pd.sum()-1

        pd0 = np.ones(self.N)/float(self.N)
        r = minimize(lambda pd: obj(pd), pd0,method='SLSQP',
                     jac=lambda pd: grad(pd),
                     bounds=Bounds(0,np.inf),
                     constraints=[{'type': 'eq', 'fun':cons}],
                     tol=1e-8)
        r['x'][r['x'] < 1e-6] = 0.0
        pd = r['x']/r['x'].sum()
        h = obj(pd)
        return pd, h

    # def reset(self, Pd):
    def reset(self):

        """
        fat is the number of attackers at time t
        lat is the rate of interaction of a, each interaction at time t happens to a defenders
        d drawn from Pd.
        fda defines the probability of Bernoulli RV of recognition of a from d
        upon such successful recognition, the harm caused by a is simply mat (the number of
        attackers of that particular a at the time instant of interaction). 
        after the response we set that particular fat and lat to zero.
        """

        self.qhat = self.Qa

        ## sample na attackers from Qa
        self.qa1 = np.random.multinomial(self.na, self.Qa)
        ## put them in the correct discrete bins
        self.qa = np.repeat(np.arange(self.N), self.qa1)

        ## initialize Pd from uniform dist
        self.Pd = np.ones(self.N)/self.N
        self.pd1 = np.random.multinomial(self.nd,self.Pd)
        self.pd = np.repeat(np.arange(self.N), self.pd1)

        self.fat = np.ones(len(self.qa))
        self.lat = np.ones(len(self.qa))
        self.mat = np.zeros(len(self.qa))

        ## used to convert na to N where ii are the indeces
        qas = np.tile(self.qa, (self.N,1))
        oo = np.arange(self.N)
        oo = np.tile(oo, (self.na,1)).T
        self.ii = (qas==oo)

        ## list of costs for each time step
        self.c = []
        self.t = 0.
        self.count_done = 0

        ## reset KF stats
        self.mu_belief_m = np.ones(self.na)/self.na
        self.cov_belief = np.eye(self.na)*0.5
        ## reset qhat beliefs
        self.qhat = np.ones(self.N)/self.N
        self.mu_m = self.qhat

        obs = self.qhat

        return obs

    def step(self,act):
        done = False
        self.t += 1.

        """
        a are the antigens for which encounter happened (1=happened, 0=not)
        """
        a = (np.random.poisson(self.lat) > 0).astype(int);
        self.mat += a
        idx = np.nonzero(a)[0]

        """
        qa is a list of all attackers present in the environment,
        a is dot{m}at the rate of the mat, i.e. attackers that have an interaction
        so we can use qa[idx] to select the indices of the attackers for which interaction happened
        d are the defenders sampled that have randomly interacted with an attacker
        """

        """
        renormalize actions centered around 0. then X_p = X+pi 
        we sample encounter from pd (available def)
        put actions from continuous [-1,1] to bins of -1,0,1
        """
        act= np.round(act).astype(int)
        self.pd += act
        ## wrap around perimeter
        self.pd[self.pd<0] = self.N-1
        self.pd[self.pd>=self.N] = 0
        self.pd = np.sort(self.pd)

        """ convert pd (size=nd, with indeces of the discrete bins)
        to a bin count of how many def in each bin with min N bins
        minlength will make sure even if a bin is missing it puts a 0
        then normalize to get distribution Pd (size N)
        """
        Pd = np.bincount(self.pd, minlength=self.N)
        self.Pd = Pd/Pd.sum()

        ## for nd != na, adjust idx of attackers interactions
        ## if num attackers sampled > number of def
        if a.sum() > self.nd:
            idx = np.random.choice(idx, self.nd, replace=False)
            idx.sort()
            d = np.random.multinomial(self.nd, self.Pd)
        ## else sample from attacker interacted with from Pd
        ## extra defenders does not help
        else:
            d = np.random.multinomial(a.sum(), self.Pd)

        d = np.repeat(np.arange(self.N), d)
        np.random.shuffle(d)

        """
        qa is a list of all antigens present in the environment,
        a are the antigens for which encounter happened (1=happened, 0=not)
        so we can use qa[idx] to select the indices of the antigens for which encounter happened
        """

        p=self.fda[d, self.qa[idx]].reshape(-1)

        recog = (np.random.random(len(p)) < p).astype(int)
        response = np.zeros(len(self.qa))
        response[idx[recog==1]] = 1

        self.fat[response > 0] = 0
        self.lat[response > 0] = 0

        self.fat += (self.fat >0)*self.nu_a
        self.lat += (self.lat >0)*self.nup_a

        """from mat calculate a remaining attacker dist
        mat is a expected number of encounters before recognition
        of size na. this dist describes the att that are most harmful
        """
        # xx = np.tile(self.mat, (self.N,1))
        # qhat = np.sum(xx*self.ii, axis=1)
        # # qhat = self.conv(qhat/qhat.sum());
        # self.qhat = qhat/qhat.sum()

        """Kalman Filter: Belief update of Qa given observations
        KF is an estimate of all na antigens
        """

        # C is an interaction of antigen, dot{m}at, a bool
        C = np.copy(a)

        var = 0.5
        z = np.random.normal(0,var)
        ## observation noise
        obs_noise = np.ones(self.na)* 2 * z
        obs_noise[C==1] = z  
        observation = C + obs_noise
        observation[response==1] = self.mat[response==1] +z
        ## observation variance noise
        Q = np.ones(self.na) * 4 * var**2
        Q[C==1] = var**2

        ## KF observation update
        K = self.cov_belief @ C.T / (C @ self.cov_belief @ C.T + Q)
        mu_belief = self.mu_belief_m + K*(observation - C*self.mu_belief_m)
        self.cov_belief = (np.eye(self.na) - np.outer(K,C)) @ self.cov_belief

        """We now have distributed observations of each type
        We want combine these multiple observations to create a single estimate
        of each type (size N) to produce a centralized KF estimate of qhat
        """
        ww = np.tile(mu_belief, (self.N,1))
        # mean of Nxna matrix -> N
        ss = np.sum(ww * self.ii, axis=1)
        mu = np.divide(ss, self.qa1, out=np.zeros_like(ss), where=self.qa1!=0)
        
        """ reocover Qa_hat from state
        original: Ma(t) = Ma(t-1)e^(nu)
        nu = log(ma(t)/ma(t-1))
        qhat = Ma(0) = Ma(t)e^(-nu*t)
        """
        nu = self.mu_m / (mu+1e-6)
        mao = mu*nu**self.t

        ## in case ma(0) is negative
        mao = np.maximum(mao, np.zeros(self.N))
        self.qhat = mao/mao.sum()
        self.qhat[np.isnan(self.qhat)] = 1/self.N
        self.qhat = self.qhat/self.qhat.sum()

        ## update prev mu with mu
        self.mu_m = mu
        self.mu_belief_m = mu_belief

        rew = -a.mean()     


        if ((self.fat>0).sum() == 0):# or t == T-1:
            self.c.append( self.Fm(self.mat).mean())
            self.count_done +=1
            # if self.count_done > 25:
                # print('done')
            done = True

        info = {}

        return self.qhat, rew, done, info

    def render(self):
        plt.clf()
        plt.plot(self.x, self.Qa, label='$Q_a$')
        plt.plot(self.x, self.qhat, label='$\hat{Q}_a$')
        plt.plot(self.x, self.Pd, label='$P_d$')
        plt.legend()
        plt.draw()
        plt.pause(0.001)

def normalize(sample):
    return sample/sample.sum()

def test(env, policy=None, render=False):
    if render:
        fig = plt.figure()

    eps_ret = []
    eps_eplen = []

    for i in range(5):
        done = False
        ep_len = 0.
        ep_ret = 0.
        
        obs = env.reset()


        while not done:
            if render:
                env.render()

            action = env.action_space.sample()
            obs, rew, done, info = env.step(action)

            ep_ret += rew
            ep_len += 1 

        eps_ret.append(ep_ret)
        print("eps_ret", ep_ret)
        eps_eplen.append(ep_len)

    avg_ret = np.mean(eps_ret)
    avg_eplen = np.mean(eps_eplen)
    print("avg_ret", avg_ret)
    print("avg_eplen", avg_eplen)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', type=int, default=0)
    args = parser.parse_args()

    env = InteractionProblem(100,100)
    test(env, render=args.render)


