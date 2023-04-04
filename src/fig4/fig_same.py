import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


"""
Written by Christopher Hsu

Plot and save figure 4 left
Na = Nd
"""


src_dir = './limits/data'
w = pickle.load(open(os.path.join(src_dir, 'w_same.pkl'),'rb'))

Nas = w['Nas']
costs = w['costs']
stdev = w['stdev']
acost = w['harm']


plt.rcParams['figure.figsize'] = [9,8]
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

plt.plot(np.arange(max(Nas)), np.ones(max(Nas))*acost, label='Analytical Harm', linewidth=3.0)
ax.scatter(Nas,costs, color='sandybrown', label='Empirical Harm')


for ii in range(len(Nas)):
    minerror = max(costs[ii]-stdev[ii], 0)
    ax.plot([Nas[ii], Nas[ii]], [costs[ii]+stdev[ii], minerror], marker="_", color='sandybrown',alpha=0.4)

ax.set_xlabel("# of agents per team")
ax.set_ylabel("Harm")

plt.legend()
plt.tight_layout()
sns.despine(ax=ax)

save_dir = './fig4/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join('/home/chris/Documents/Model4PD2022/figures','limits_same.pdf'),bbox_inches='tight')

plt.show()