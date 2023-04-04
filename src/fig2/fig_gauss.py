import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


"""
Written by Christopher Hsu

Plot and save figure 2
"""


src_dir = './fig2/data/'
s = pickle.load(open(os.path.join(src_dir, 'sim_sigs.pkl'),'rb'))
w = pickle.load(open(os.path.join(src_dir, 'w_react_sigs.pkl'),'rb'))

sigs = s['sigs']
costs = s['costs']
acosts = s['acosts']
stdev = s['stdev']
h = s['h']
diversity = s['diversity']
pds = s['pds']

sns.set_theme()
fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=0.7*fsz)
plt.rc('figure', titlesize=fsz)
plt.rc('pdf', fonttype=42)
sns.set_style("ticks", rc={"axes.grid":True})


## fig 2 left
fig, ax = plt.subplots(figsize=(8,8))
ax.plot(w['x'],w['qa'],label='$\sigma_Q=0.1$', linewidth=4.0);
ax.plot(w['x'],pds[10],label='$\sigma_P=0.05$', linewidth=4.0);
ax.plot(w['x'],pds[27],label='$\sigma_P=0.13$', linewidth=4.0);
ax.plot(w['x'],pds[-1],label='$\sigma_P=0.2$', linewidth=4.0);
plt.legend();
ax.set_xlabel('State (x)')
ax.set_ylabel('Probability')
plt.tight_layout()
sns.despine(ax=ax)


## fig 2 right
fig2, ax2 = plt.subplots(figsize=(8,8))
color = 'tab:blue'
ax2.set_xlabel('Cross-Reactivity Bandwidth $\sigma$')
ax2.scatter(sigs, costs, color=color, label='Empirical Harm')

for ii in range(len(sigs)):
    minerror = max(costs[ii]-stdev[ii], 0)
    ax2.plot([sigs[ii], sigs[ii]], [costs[ii]+stdev[ii], minerror], marker="_", color=color, alpha=0.5, linewidth=2.0)
ax2.set_ylabel('Empirical Harm', color=color)
ax2.tick_params(axis='y', labelcolor=color)

ax3 = ax2.twinx()
color = 'tab:orange'
ax3.plot(sigs, diversity, label='', color=color, linewidth=4.0)
ax3.set_ylabel('Diversity of Defense ($||P_d^*||_0$)', color=color)
ax3.tick_params(axis='y', labelcolor=color)

fig2.tight_layout()
sns.despine(ax=ax2, right=False)
sns.despine(ax=ax3, right=False)

save_dir = './fig2/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join(save_dir,'gauss_probs.pdf'),bbox_inches='tight')
fig2.savefig(os.path.join(save_dir,'gauss_harms.pdf'),bbox_inches='tight')

plt.show()