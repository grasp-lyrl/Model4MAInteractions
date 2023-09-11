import os, time, datetime, pickle, random
import os.path as osp
import numpy as np
import torch
from envs.interactGym import InteractionProblem
from utils.mpi_tools import mpi_fork
from utils.logx import EpochLogger
from ipdb import set_trace as st
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_theme()
fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=fsz)
plt.rc('figure', titlesize=fsz)
plt.rc('pdf', fonttype=42)
sns.set_style("ticks", rc={"axes.grid":True})

def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
setup(42)

def load_pytorch_policy(fpath, itr, deterministic=False):
    """ Load a pytorch policy saved with Spinning Up Logger."""
    
    fname = osp.join(fpath, 'pyt_save', 'model'+itr+'.pt')
    print('\n\nLoading from %s.\n\n'%fname)

    model = torch.load(fname)

    # make function for producing an action given a single state
    def get_action(x):
        with torch.no_grad():
            x = torch.as_tensor(x, dtype=torch.float32)
            action = model.act(x)
        return action

    return get_action

def wasserstein_distance(p1,p2):
    ## earth movers distance in 1D, 1-norm of the CDFs
    return np.linalg.norm(np.cumsum(p1)-np.cumsum(p2), ord=1)

def run_policy(env, get_action, num_episodes=100, render=True, max_ep_len=500, save=1):

    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    Pdstar, opt_h = env.calc_pd(env.Qa)
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    nd = np.ones(env.N)/env.N
    nd_tot = 110
    nd *= nd_tot

    Nds=[]
    Pds = []
    qhats = []
    rl_h = []
    while n < num_episodes:
        ## problem runs until done then n+=1

        if render:
            env.render()
            time.sleep(1e-3)

        a = get_action(o)

        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)

            rl_h.append(np.sum(env.qhat*env.Fbar(env.conv(env.Pd))))
            Pds.append(env.Pd)
            endPd = env.Pd

            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    Pdstar, opt_h = env.calc_pd(env.Qa)
    Pds = np.asarray(Pds)
    Pd = np.sum(Pds,axis=0)
    Pd = Pd/Pd.sum()

    avg_rl_h = np.sum(env.Qa*env.Fbar(env.conv(Pd)))
    end_rl_h = np.sum(env.Qa*env.Fbar(env.conv(endPd)))

    s = dict(x=env.x,
             Qa=env.Qa,
             Pds=Pds,
             avgPd=Pd,
             endPd=endPd,
             Nds=Nds,
             Pdstar=Pdstar,
             opt_h=opt_h,
             avg_rl_h=avg_rl_h,
            )

    print('opt harm', opt_h, 'avg PD sac harm', avg_rl_h, 'one sac harm', end_rl_h)

    fig, ax = plt.subplots(figsize=(8,8))

    # plt.plot(env.x, env.Qa, label='$Q_a$', linewidth=2.0)
    plt.plot(env.x, Pdstar, label='$P_d^*(Q_a)$', linewidth=4.0)
    plt.plot(env.x, endPd, label='$P_d(\hat{Q}_a)$', linewidth=4.0)
    plt.xlabel('State (x)')
    plt.ylabel('Probability')
    plt.legend()
    plt.tight_layout()
    sns.despine(ax=ax)    

    if save:
        save_dir = './results'
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        pickle.dump(s, open(os.path.join(save_dir,'sim.pkl'), 'wb'))
        fig.savefig(os.path.join(save_dir,'sacInteraction.pdf'),bbox_inches='tight')
    plt.show()

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()



if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='InteractionProblem')
    parser.add_argument('--mode', choices=['train','test'], default='train')
    parser.add_argument('--hid', type=int, default=64)
    parser.add_argument('--l', type=int, default=2)
    parser.add_argument('--gamma', type=float, default=0.99)
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--cpu', type=int, default=2)
    parser.add_argument('--steps', type=int, default=10000)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='sac')
    parser.add_argument('--log_dir', type=str, default='./results')
    parser.add_argument('--save_name', type=str, default='data')
    parser.add_argument('--fpath', type=str, default='./results/sac/sac_s0')
    parser.add_argument('--num_test_ep', type=int, default=5)
    parser.add_argument('--test_render', type=int, default=1)
    parser.add_argument('--na', type=int, default=100)
    parser.add_argument('--nd', type=int, default=100)
    args = parser.parse_args()

    # mpi_fork(args.cpu)  # run parallel code with mpi

    from utils.run_utils import setup_logger_kwargs
    # save_dir = args.exp_name+'_%da%dd_'%(args.na,args.nd)+args.save_name
    save_dir = args.exp_name
    logger_kwargs = setup_logger_kwargs(save_dir, args.seed, args.log_dir)

    env = InteractionProblem(args.na,args.nd)

    env_fn = lambda: env

    if args.mode == 'train':
        if args.exp_name == 'sac':
            from algos.pytorch.sac.sac import sac
            import algos.pytorch.sac.core as core        

            sac(env_fn, actor_critic=core.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                logger_kwargs=logger_kwargs)

        elif args.exp_name == 'td3':
            from algos.pytorch.td3.td3 import td3
            import algos.pytorch.td3.core as core

            td3(env_fn, actor_critic=core.MLPActorCritic,
                ac_kwargs=dict(hidden_sizes=[args.hid]*args.l), 
                gamma=args.gamma, seed=args.seed, epochs=args.epochs,
                logger_kwargs=logger_kwargs)

    else:
        get_action = load_pytorch_policy(args.fpath,itr='',deterministic=True)
        run_policy(env,get_action,args.num_test_ep, args.test_render)
