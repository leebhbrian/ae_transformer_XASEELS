import matplotlib.pyplot as plt
import numpy as np


def plot_parity(res,fname,min_val=-0.05,max_val=2.05):
    fig, ax1=plt.subplots(figsize=(6,5))
    plt.scatter(res[:,0], res[:,1],s=20,marker='o',color='b')
    plt.plot([0, max_val], [0, max_val], 'k--')  # black dashed line
    plt.xlabel("Oxidation, True",fontsize=16)
    plt.ylabel("Oxidation, Predicted",fontsize=16)
    plt.gca().set_aspect('equal', adjustable='box')
    plt.xlim(min_val, max_val)
    plt.ylim(min_val, max_val)
    plt.tight_layout()
    plt.savefig(f'{fname}.png', dpi=300)