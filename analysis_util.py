import matplotlib.pyplot as plt
import numpy as np


def plot_parity(res,fname,min_val=0.0,max_val=2.0):
    fig, ax1=plt.subplots(figsize=(6,5))
    yerr = res[:,2] if res.shape[1] > 2 else None
    plt.errorbar(res[:,0], res[:,1], yerr=yerr, fmt='o', ms=3,
                 color='b', ecolor='b', elinewidth=1, capsize=2,linestyle='None')
    plt.plot([0, max_val], [0, max_val], 'k--')  # black dashed line
    plt.xlabel("Oxidation, True",fontsize=16)
    plt.ylabel("Oxidation, Predicted",fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig(f'{fname}', dpi=300)