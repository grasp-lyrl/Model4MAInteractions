import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm
from ipdb import set_trace as st
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
import pandas


"""
Written by Christopher Hsu

Plot and save figure

base case (not used in paper)
"""


src_dir = './fig7competition/data/base'
s = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))


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

ax.plot(w['x'],w['pd'],label='$P_d^*$', linewidth=4.0);
ax.plot(w['x'],s['Pd'],label='$P_d$', linewidth=4.0);
# ax.plot(w['x'],s['harm'], label='Harm($P_d$)', linewidth=1.5)

ax.set_xlabel('State (x)')
ax.set_ylabel('Probability')

plt.legend()
plt.tight_layout()
sns.despine(ax=ax)


fig1, ax1 = plt.subplots(figsize=(8,8))

ax1.plot(s['wass'],label='$Wass_1$($P_d^*$, $P_d$)', linewidth=1.5);

ax1.set_xlabel('Episodes')
ax1.set_ylabel('$Wass_1$($P_d^*$, $P_d$)')
plt.tight_layout()
sns.despine(ax=ax)


# fig3, ax = plt.subplots(figsize=(8,8))
# x = np.arange(len(s['c']))
# ax.plot(x, np.repeat(w['h'],len(s['c'])),label='$P_d^*(Q_a)$ Analytical');
# ax.plot(x, s['c'],label='$P_d(\hat{Q}_a,t)$ Empirical');
# plt.tight_layout()
# sns.despine(ax=ax)

# ##empirical variance plotting
# costs = s['c']
# stdev = s['C'][1]
# for ii in range(len(x)):
#     minerror = max(costs[ii]-stdev, 0)
#     ax.plot([x[ii], x[ii]], [costs[ii]+stdev, minerror], marker="_", color='sandybrown',alpha=0.3)

plt.rc('legend', fontsize=0.8*fsz)

cost = np.asarray(s['c']).reshape(4,25)
data = {'25':cost[0,:],'50':cost[1,:],'75':cost[2,:],'100':cost[3,:]}
df = pandas.DataFrame(data=data)

fig3, ax = plt.subplots(figsize=(8,8))
x = np.arange(4)
ax.plot(x, np.repeat(w['h'],4),label='$P_d^*(Q_a)$ Analytical', linewidth=5);
sns.boxplot(df, color='sandybrown')

ax.set_xlabel('Episodes')
ax.set_ylabel('Harm')
plt.legend()
plt.tight_layout()
sns.despine(ax=ax)


save_dir = './fig7competition/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join(save_dir,'comp_base.pdf'),bbox_inches='tight')
fig1.savefig(os.path.join(save_dir,'comp_base_wass.pdf'),bbox_inches='tight')
fig3.savefig(os.path.join(save_dir,'comp_base_harm.pdf'),bbox_inches='tight')

plt.show()