import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm

import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


"""
Written by Christopher Hsu

Plot and save figure 3 right
"""


src_dir = './fig3/data/nonunif'
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

ax.plot(w['x'],w['qa'],label='$Q_a$', linewidth=3.0);
ax.plot(w['x'],w['pd'],label='$P_d^*$', linewidth=3.0);
ax.plot(w['x'],s['harm'], label='$Q_a \overline{F}_a$', linewidth=4.0)

ax.set_xlabel('State (x)')
ax.set_ylabel('Probability')

plt.legend()
plt.tight_layout()
sns.despine(ax=ax)

save_dir = './fig3/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join(save_dir,'nonunif.pdf'),bbox_inches='tight')

plt.show()