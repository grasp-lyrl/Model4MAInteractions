import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm
from ipdb import set_trace as st
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


"""
Written by Christopher Hsu

Plot and save figure

base case (not used in paper)
"""


src_dir = './perimdef/data/shuffle0_control0'
s00 = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w00 = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))

src_dir = './perimdef/data/shuffle0_control1'
s01 = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w01 = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))

src_dir = './perimdef/data/shuffle1_control0'
s10 = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w10 = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))

src_dir = './perimdef/data/shuffle1_control1'
s11 = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w11 = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))


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

# fig, ax = plt.subplots(figsize=(8,8))
# ax = fig.add_subplot()
# ## plot perimeters
# ax.plot(perim_xy[:,0], perim_xy[:,1])
# ax.plot(att_xy[:,0], att_xy[:,1])

# ## plot agentss
# ax.scatter(defenders[:,0], defenders[:,1], marker='.',label='defender');
# ax.scatter(intruders[:,0], intruders[:,1], marker='.',label='intruder');

# plt.tight_layout()
# plt.legend()
captures = []
captures.append(s00['capture'])
captures.append(s01['capture'])
captures.append(s10['capture'])
captures.append(s11['capture'])
captures = np.asarray(captures).T

# fig, ax = plt.subplots(figsize=(8,8))
# ax = fig.add_subplot()

# ax.boxplot(captures)

# plt.show()



fig1, ax = plt.subplots(figsize=(8,8))

ax.plot(w10['x'],w10['qa'],label='$Q_a$', linewidth=4.0);
ax.plot(w10['x'],w10['pd'],label='$P_d^*$', linewidth=4.0);
# ax.plot(w['x'],s['harm'],label='Harm', linewidth=4.0);

ax.set_xlabel('State (x)')
ax.set_ylabel('Probability')
plt.legend()
plt.tight_layout()
sns.despine(ax=ax)

save_dir = './perimdef/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

# fig.savefig(os.path.join(save_dir,'pd.pdf'),bbox_inches='tight')
fig1.savefig(os.path.join(save_dir,'pd_shape.pdf'),bbox_inches='tight')