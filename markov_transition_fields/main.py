# Task 1: Import Required Libraries and Packages
import pandas as pd    # pip install pandas
import numpy as np     # pip install numpy
from skimage import measure as sm    # pip install scikit-image
# pip install pyts
# restart kernal after pyts install
from pyts.preprocessing.discretizer import KBinsDiscretizer
import matplotlib.pyplot as plt    # pip install matplotlib
from matplotlib import cm, colors

df = pd.read_csv("https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/ETT-small/ETTh1.csv")

ett = df.truncate(after = 999)

fig = plt.figure(figsize=(28,4))
plt.plot(ett['date'], ett['HUFL'], linewidth=1)
plt.savefig("plot_1.png")

n_bins = 10
strategy = 'quantile'
discretizer = KBinsDiscretizer(n_bins = n_bins, strategy = strategy, raise_warning = False)
X = ett['HUFL'].values.reshape(1, -1)
ett['HUFL_disc'] = discretizer.fit_transform(X)[0]

m_adj = np.zeros((10, 10))
for k in range(len(ett.index) - 1):
    # Matrix Iteration
    index = ett['HUFL_disc'][k]
    next_index = ett['HUFL_disc'][k + 1]
    m_adj[next_index][index] += 1

mtm = m_adj/m_adj.sum(axis=0)

n_t = len(ett.index)
mtf = np.zeros((n_t, n_t))

for i in range(n_t):
    for j in range(n_t):
        mtf[i, j] = mtm[ett['HUFL_disc'][i]][ett['HUFL_disc'][j]]*100

fig = plt.figure()
plt.imshow(mtf)
plt.colorbar()

plt.savefig("plot_2.png")

mtf_reduced = sm.block_reduce(mtf, block_size = (10, 10), func = np.mean)

fig = plt.figure(figsize = (8, 8))

plt.imshow(mtf_reduced)
plt.colorbar()

plt.savefig("plot_3.png")

mtf_diag = [mtf_reduced[i][i] for i in range(len(mtf_reduced))]
fig, ax = plt.subplots(figsize = (28, 4))
norm = colors.Normalize(vmin=np.min(mtf_diag), vmax=np.max(mtf_diag))
cmap = cm.viridis
for i in range(0, n_t, 10):
    ax.plot(ett['date'][i:i+10+1], ett['HUFL_disc'][i:i+10+1], c = cmap(norm(mtf_diag[int(i/10)])))

plt.colorbar(cm.ScalarMappable(norm=norm, cmap=cmap), ax = ax)
plt.savefig("plot_4.png")
