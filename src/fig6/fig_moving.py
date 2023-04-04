import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


"""
Written by Christopher Hsu

Plot and save figure 6
"""

src_dir = './fig6/data/moving'

s = pickle.load(open(os.path.join(src_dir, 'sim_moving.pkl'),'rb'))
w = pickle.load(open(os.path.join(src_dir, 'w_moving.pkl'),'rb'))

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


fig, ax = plt.subplots(figsize=(8,8))

ax.plot(w['x'],w['qa'],label='$Q_a$', linewidth=4.0);
ax.plot(w['x'],s['qhat'],label='$\hat{Q}_a$', linewidth=4.0);
ax.set_ylabel('Probability')
ax.set_xlabel('State (x)')

plt.legend()
plt.tight_layout()
sns.despine(ax=ax)


fig1, ax = plt.subplots(figsize=(8,8))

ax.plot(w['x'],w['pd'],label='$P_d^*(Q_a)$', linewidth=4.0);
ax.plot(w['x'],s['Pd'],label='$P_d(\hat{Q}_a)$', linewidth=4.0);
ax.set_ylabel('Probability')
ax.set_xlabel('State (x)')
plt.legend()
plt.tight_layout()
sns.despine(ax=ax)


fig2, ax = plt.subplots(figsize=(8,8))

ax.plot(np.arange(len(s['qawds'])),s['qawds'],label='$W_1$($Q_a$, $\hat{Q}_a$)',linewidth=3.0);
ax.plot(np.arange(len(s['pdwds'])),s['pdwds'],label='$W_1$($P_d^*$, $P_d$)',linewidth=3.0);

ax.set_xlabel('Episodes')
ax.set_ylabel('$W_1$ Metric')

plt.legend()
plt.tight_layout()
sns.despine(ax=ax)



plt.rc('legend', fontsize=0.8*fsz)
fig3, ax = plt.subplots(figsize=(8,8))
x = np.arange(len(s['c']))
ax.plot(x, np.repeat(w['h'],len(s['c'])),label='$P_d^*(Q_a)$ Analytical',linewidth=3.0);
ax.plot(x, s['c'],label='$P_d(\hat{Q}_a,t)$ Empirical',linewidth=3.0);

##empirical variance plotting
costs = s['c']
stdev = s['C'][1]
for ii in range(len(x)):
    minerror = max(costs[ii]-stdev, 0)
    ax.plot([x[ii], x[ii]], [costs[ii]+stdev, minerror], marker="_", color='sandybrown',alpha=0.3)

ax.set_xlabel('Episodes')
ax.set_ylabel('Harm')
plt.legend()
plt.tight_layout()
sns.despine(ax=ax)


save_dir = './fig6/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join(save_dir,'f6-qa.pdf'),bbox_inches='tight')
fig1.savefig(os.path.join(save_dir,'f6-pd.pdf'),bbox_inches='tight')
fig2.savefig(os.path.join(save_dir,'f6-wd.pdf'),bbox_inches='tight')
fig3.savefig(os.path.join(save_dir,'f6-harms.pdf'),bbox_inches='tight')

plt.show()