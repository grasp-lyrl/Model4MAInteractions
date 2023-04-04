import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


"""
Written by Christopher Hsu

Plot and save figure 3 middle
"""

src_dir = './fig3/data/wass'
s = pickle.load(open(os.path.join(src_dir, 'sim_wass.pkl'),'rb'))
wds = np.asarray(s['wds'])
acosts = np.asarray(s['acosts'])
costs = np.asarray(s['costs'])
stdevs = np.asarray(s['stdevs'])


sns.set_theme()
fsz = 32
plt.rc('font', size=fsz)
plt.rc('axes', titlesize=fsz)
plt.rc('axes', labelsize=fsz)
plt.rc('xtick', labelsize=fsz)
plt.rc('ytick', labelsize=fsz)
plt.rc('legend', fontsize=0.8*fsz)
plt.rc('figure', titlesize=fsz)
plt.rc('pdf', fonttype=42)
sns.set_style("ticks", rc={"axes.grid":True})

fig, ax = plt.subplots(figsize=(8,8))

ax = sns.regplot(x=wds, y=acosts, data=acosts, lowess=True, label='Analytical Harm',scatter_kws={'alpha': 1,'s':4})
ax = sns.regplot(x=wds, y=costs, data=costs, lowess=True, label='Empirical Harm',scatter_kws={'alpha': 0.3,'s':4})

## error bar
for ii in range(len(wds)):
    minerror = max(costs[ii]-stdevs[ii], 0)
    ax.plot([wds[ii], wds[ii]], [costs[ii]+stdevs[ii], minerror], marker="_", color='sandybrown',alpha=0.3, linewidth=1.5)

ax.set_xlabel('$W_1$($P_d$, $P_d^*$)')
ax.set_ylabel('Harm')
ax.set_ylim(0,200)

plt.legend()

plt.tight_layout()
sns.despine(ax=ax)

save_dir = './fig3/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join(save_dir,'wass.pdf'),bbox_inches='tight')

plt.show()