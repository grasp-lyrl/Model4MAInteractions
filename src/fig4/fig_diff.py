import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm


"""
Written by Christopher Hsu

Plot and save figure 4 right
Na != Nd
"""


src_dir = './fig4/data'
w = pickle.load(open(os.path.join(src_dir, 'w_diff.pkl'),'rb'))

Nas = w['Nas']
Nds = w['Nds']
costs = w['costs']
stdev = w['stdev']


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


fig1 = plt.figure()
ax = fig1.add_subplot()

cmap = sns.color_palette("rocket_r", as_cmap=True)
X = np.arange(1,100,5)
Y = np.arange(1,100,5)
cp = ax.contourf(X,Y,costs.reshape(20,20).T,levels=[5,10,20,40,80,160],cmap=cmap, norm=LogNorm())

ax.set_xlabel("$\sum_a N_a$ (# of Attackers)")
ax.set_ylabel("$\sum_d N_d$ (# of Defenders)")

cbar = fig1.colorbar(cp)
cbar.ax.set_ylabel('Mean Harm')

save_dir = './fig4/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig1.savefig(os.path.join(save_dir,'limits_diff.pdf'),bbox_inches='tight')

plt.show()