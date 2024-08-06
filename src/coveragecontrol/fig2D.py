import numpy as np
import os, sys, ipdb, datetime, click, pickle #tqdm
from ipdb import set_trace as st
import seaborn as sns
sns.set_context('talk')
import matplotlib.pyplot as plt


"""
Written by Christopher Hsu

Plot and save figure

2D plotting for coverage control
"""


src_dir = './coveragecontrol/data/base4'
s4 = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w4 = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))

src_dir = './coveragecontrol/data/base8'
s8 = pickle.load(open(os.path.join(src_dir, 'sim.pkl'),'rb'))
w8 = pickle.load(open(os.path.join(src_dir, 'w_react.pkl'),'rb'))


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


N = w8['N']
nn = int(np.sqrt(N))

cmap = sns.color_palette("viridis", as_cmap=True)

fig, ax = plt.subplots(figsize=(8,8))
im = ax.imshow(w8['qa'].reshape(nn,nn),vmin=0.0,vmax=0.02,label='$Q_a$',cmap=cmap,extent=[0,1,0,1])
ax.set_xlabel('State ($x_1$)')
ax.set_ylabel('State ($x_2$)')
cbar = plt.colorbar(im)
cbar.ax.set_ylabel('Probability')


cmap = sns.color_palette("rocket_r", as_cmap=True)

import random
def setup(seed):
    np.random.seed(seed)
    random.seed(seed)
setup(42)

fig1, ax1 = plt.subplots(figsize=(8,8))

# N_d = 10
# # init = np.random.randint(0,nn,(N_d,2))
# init = np.sort(np.random.randint(0,400,N_d))
# rows_i = init%nn
# cols_i = init//nn

# d = np.random.multinomial(N_d, w8['pd'])
# d = np.repeat(np.arange(N), d)
# rows = d%nn
# cols = d//nn
# ## plot agentss
# ax1.scatter(rows, cols, marker='o',label='defender');
# ax1.scatter(rows_i, cols_i, marker='o',label='init');

im1 = ax1.imshow(w8['pd'].reshape(nn,nn),vmin=0.0,vmax=0.1,label='$P_d$',cmap=cmap,extent=[0,1,0,1])

# for ii in range(N_d):
#     ax1.plot((rows_i[ii],rows[ii]), (cols_i[ii],cols[ii]))
ax1.set_xlabel('State ($x_1$)')
ax1.set_ylabel('State ($x_2$)')
cbar = plt.colorbar(im1)
cbar.ax.set_ylabel('Probability')



fig2, ax2 = plt.subplots(figsize=(8,8))

im2 = ax2.imshow(w4['pd'].reshape(nn,nn),vmin=0.0,vmax=0.02,label='$P_d$',cmap=cmap,extent=[0,1,0,1])

ax2.set_xlabel('State ($x_1$)')
ax2.set_ylabel('State ($x_2$)')
cbar = plt.colorbar(im2)
cbar.ax.set_ylabel('Probability')



save_dir = './coveragecontrol/figures'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

fig.savefig(os.path.join(save_dir,'qa2d.pdf'),bbox_inches='tight')
fig1.savefig(os.path.join(save_dir,'sig8_shape.pdf'),bbox_inches='tight')
fig2.savefig(os.path.join(save_dir,'sig4_shape.pdf'),bbox_inches='tight')

plt.show()