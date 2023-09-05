import numpy as np
import scipy as sp
from scipy.optimize import minimize, Bounds

import os, sys, pdb, random, datetime, pickle, argparse
from ipdb import set_trace as st

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
from matplotlib import animation
plt.rcParams['figure.figsize'] = [8,8]

sns.set_theme()
fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=0.6*fsz)
plt.rc('figure', titlesize=fsz)
plt.rc('pdf', fonttype=42)
sns.set_style("ticks", rc={"axes.grid":True})


"""
Written by Christopher Hsu

Given Qa, find Pd and simulate the perimter defense game
"""

def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
setup(42)

# gaussian fra bandwidth
sigma=0.05
D = 0.1*sigma
beta = 0.2

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
# Fbar = lambda pa: sp.special.gamma(1+alpha)/pa**alpha
# dFbar = lambda pa: -alpha*sp.special.gamma(1+alpha)/pa**(1+alpha)
# Fm = lambda m: m**alpha
Fbar = lambda pa: beta/(beta+pa)
dFbar = lambda pa: -beta/(beta+pa)**2
Fm = lambda m: 1-np.exp(-beta*m)


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

def polar2cart(polar):  ##[num_agents][r, theta]
    polar = np.array(polar)
    if polar.ndim == 2:
        cart = np.zeros((polar.shape[0], 2))
        cart[:, 0] = polar[:, 0]*np.cos(polar[:,1])
        cart[:, 1] = polar[:, 0]*np.sin(polar[:,1])
    else:
        cart = np.zeros(2)
        cart[0] = polar[0]*np.cos(polar[1])
        cart[1] = polar[0]*np.sin(polar[1])
    return cart

def cartesian2polar(xy): ## theta in [-pi,pi]
    if xy.ndim==2:
        r = np.sqrt(np.sum(xy**2,axis=1))
        theta = np.arctan2(xy[:,1], xy[:,0])
    else:
        r = np.sqrt(np.sum(xy**2))
        theta = np.arctan2(xy[1], xy[0])
    return np.vstack((r,theta)).T

def create_perimeter(pts):
    x = []
    y = []
    dist = np.zeros(pts.shape[0]-1)
    for ii in range(pts.shape[0]-1):
        dist[ii] = np.linalg.norm(pts[ii,:]-pts[ii+1,:])
    num_N_per_line = np.round(N*(dist/dist.sum()))

    for ii in range(pts.shape[0]-1):
        x = np.concatenate((x, np.linspace(pts[ii,0],pts[ii+1,0],int(num_N_per_line[ii]), endpoint=False)))
        y = np.concatenate((y, np.linspace(pts[ii,1],pts[ii+1,1],int(num_N_per_line[ii]), endpoint=False)))
    perim_xy = np.vstack((x,y)).T

    ## create attacker init perim 1 unit away
    polar = cartesian2polar(perim_xy)
    polar[:,0] += 1
    att_xy = polar2cart(polar)

    # fig, ax = plt.subplots()
    # ax.plot(xy[:,0],xy[:,1])
    # ax.plot(att_xy[:,0],att_xy[:,1])
    # plt.show()
    return perim_xy, att_xy


def simulate(w,e=100,na=int(100),debug=False):
    if args.render:
        fig = plt.figure()
        fig.clf()

        if args.record:
            fname = os.path.join('./perimdef/data/figures', 'eval_PerimDef_shuffle%d_control%d.mp4'%(args.shuffle,args.control))
            moviewriter = animation.FFMpegWriter(codec='libx264')
            moviewriter.setup(fig,outfile=fname,dpi=80)
    """
    sample na attackers from Qa. samples na def from Pd.
    record the ratio of recognized/total 
    """
    N = w['N']
    c = []
    capture = []
    mat_list = []
    ## define perimeter
    ## we want to also map to arbitrary polygon based on arc length
    pts = np.array([[-0.5,0],
                    [-0.5,0.5],
                    [0,1],
                    [1,1],
                    [0.5,0],
                    [1,-1],
                    [0,-1],
                    [-0.5,-0.5],
                    [-0.5,0]])
    perim_xy, att_xy = create_perimeter(pts)

    for ee in range(e):
        ## sample na attackers from Qa
        qa = np.random.multinomial(na, w['qa'])
        qa = np.repeat(np.arange(N), qa)
        ## get bin indexs for agents
        qas = np.tile(qa, (N,1))
        oo = np.arange(N)
        oo = np.tile(oo, (na,1)).T
        ii = (qas==oo)

        fat = np.ones(len(qa)); lat = np.ones(len(qa)); mat = np.zeros(len(qa))

        ## interacting defenders
        Pd = w['pd']
        # Pd = np.ones(N)/N
        # Pd =w['qa']

        d = np.random.multinomial(na, Pd)
        d = np.repeat(np.arange(N), d)
        # random assigment
        # np.random.shuffle(d)

        ## initialize intruders and defenders in cartesian
        intruders_init = att_xy[qa]

        ## attacker velocity
        v_a = -0.05

        T = int(1/abs(v_a))
        for tt in range(T):
            
            a = (np.random.poisson(lat) > 0).astype(int);
            mat += a
            idx = np.nonzero(a)[0]
   
            # d = np.random.multinomial(a.sum(), Pd)
            # d = np.repeat(np.arange(N), d)
            ## random assigment
            if args.shuffle:
                np.random.shuffle(d)

            response = np.zeros(len(qa))
            ## def already assigned an intruder
            ## update positions
            ## convert intruder cart to polar then move a radial velocity inwards
            polar = cartesian2polar(intruders_init)
            polar[:,0] += v_a
            intruders_init = polar2cart(polar)

            ## move by bin
            ## 1 bin is D = 0.005 therefore 2 bins is 0.01 which is similar to the cross rxn bandwidth
            ## if agent 0 is assigned 199 theyll go the long way around thats why random is bad now and theres no errors
            if args.control:
                control = limit_vel(qa[idx] - d[idx], max_v=1)
            else:
                control = 0
            d[idx] += control
            defenders = perim_xy[d[idx]]

            ## position capture
            ## recognition if def is within eps dist from att
            # eps = np.ones(na)*D
            # dist = np.linalg.norm(intruders - defenders, axis=1) 
            # recog = (dist <= eps).astype(int)

            ## cross reactivity capture
            p=w['fda'][d[idx], qa[idx]].reshape(-1)
            ## bernoulli rv of interaction (1=recognition, 0=unsuccessful)
            recog = (np.random.random(len(p)) < p).astype(int)

            ## record captures
            response[idx[recog==1]] = 1

            lat[response > 0] = 0
            lat += (lat >0)*nup_a

            fat[response > 0] = 0
            intruders = intruders_init[fat==1]
            
            if args.render:
                fig.clf()
                ax1 = fig.add_subplot()
                # ax.plot(w['x'], w['qa'],label='$Q_a$', linewidth=2.);
                # ax.plot(w['x'], w['pd'],label='$P_{d,u}^*$', linewidth=2.);

                ## plot perimeters
                ax1.plot(att_xy[:,0], att_xy[:,1], linewidth=3.0, label='att init')
                ax1.plot(perim_xy[:,0], perim_xy[:,1], linewidth=3.0, label='perimeter')

                ## plot agentss
                ax1.scatter(intruders[:,0], intruders[:,1], marker='o',label='attacker');
                ax1.scatter(defenders[:,0], defenders[:,1], marker='o',label='defender');

                ax1.set_xlabel('State ($x_1$)')
                ax1.set_ylabel('State ($x_2$)')
                plt.tight_layout()
                plt.legend()
                plt.draw()
                plt.pause(0.001)

                if args.record:
                    moviewriter.grab_frame()
                # if tt == 1 and ee==0:
                #     save_dir = './perimdef/data/figures'
                #     if not os.path.exists(save_dir):
                #         os.makedirs(save_dir)
                #     fig.savefig(os.path.join(save_dir,'pdgame.pdf'),bbox_inches='tight')
                    # plt.show()

            if fat.sum() == 0:
                break

        c.append(Fm(mat).mean())
        capture.append(1-fat.sum()/na)
        if args.render:
            print(capture[-1], w['h'], Fm(mat).mean(), tt)

        if e > 10 and (ee % (e//10) == 0):
            print('[%04d]'%ee)


        ## episode harm distribution from na attackers to N bins
        xx = np.tile(fat, (N,1))
        mm = np.sum(xx*ii, axis=1)
        mm = mm/N
        mm = mm/mm.sum()
        mat_list.append(mm)
    ## average over all episodes harm and smooth with conv
    mm = np.asarray(mat_list)
    mm = np.sum(mm,axis=0)
    mm = mm/mm.sum()
    # mm = conv(mm)/conv(mm).sum()

    print(np.mean(c), np.std(c))

    return dict(Qa=w['qa'],
                Pd=Pd,
                harm=mm,
                perim_xy=perim_xy,
                att_xy=att_xy,
                capture=capture,
                Capture=(np.mean(capture),np.std(capture)),
                c=c,
                C=(np.mean(c), np.std(c)))


if __name__ == '__main__':

    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--save', type=int, default=1)
    parser.add_argument('--control', type=int, default=0)
    parser.add_argument('--shuffle', type=int, default=0)
    parser.add_argument('--render', type=int, default=1)
    parser.add_argument('--record', type=int, default=0)
    args = parser.parse_args()

    ## sample Qa from lognormal
    sig = (np.log(kappa**2 + 1))**.5
    Qa = np.random.lognormal(mean=0,sigma=sig,size=N)
    Qa=Qa/Qa.sum();

    w = cross_react(Qa,fda)
    # print(w['h'])
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
        save_dir = './data/shuffle%d_control%d'%(args.shuffle,args.control)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # fig.savefig(os.path.join(save_dir,'pd.png'))
        pickle.dump(s, open(os.path.join(save_dir,'sim.pkl'), 'wb'))
        pickle.dump(w, open(os.path.join(save_dir,'w_react.pkl'), 'wb'))
    
    plt.show()