import numpy as np
from scipy import stats
from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns

def scatter_subs_sim(var1, var2, xlabel=None, y_label=None, plot_path=None,
                     color='tab:blue'):
    n = var2.shape[0]
    correls = np.zeros(n)
    for sub_i in range(n):
        correls[sub_i], _ = stats.pearsonr(
            var1, var2[sub_i, :])
    r_mean, r_std = correls.mean(), correls.std()
    df = {'x': np.concatenate([var1 for _ in range(n)]),
          'y': np.concatenate([var2[i, :] for i in range(n)]),
          's_id': np.concatenate([np.ones(len(var1)) * i for i in range(n)])}
    df = pd.DataFrame(df)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    sns.lineplot(data=df, x="x", y="y", color=color)
    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(y_label, fontsize=20)
    txt = "r={:.3f}(".format(r_mean) + r'$\pm$' + "{:.3f})".format(r_std)
    bbox_props = dict(boxstyle="round", fc="w", ec="0.5", alpha=0.6)
    ax.text(x=.05, y=.9, s=txt, size=18,
            transform=ax.transAxes, bbox=bbox_props)
    ax.tick_params(axis='both', which='major', labelsize=16)
    ax.tick_params(axis='both', which='minor', labelsize=16)
    plt.subplots_adjust(left=.22, bottom=.2, right=.90)
    # plt.tight_layout()
    if not plot_path is None:
        plt.savefig(plot_path, dpi=600)
    plt.show()
    return correls
