import matplotlib.pyplot as plt
import numpy as np


plt.rcParams.update({
    "font.family": "serif",
    "font.size": 11
})


fig, axes = plt.subplots(2, 2, figsize=(14, 15))

# --- (A) GBoost ---
ax = axes[0, 0]
x_gb = np.linspace(10, 300, 20)
train_gb = 0.68 - 0.2*np.exp(-x_gb/50)
cv_gb = 0.58 - 0.15*np.exp(-x_gb/40)
ax.plot(x_gb, train_gb, 'o-', label='Train $R^2$')
ax.plot(x_gb, cv_gb, 's--', label='CV $R^2$')
ax.fill_between(x_gb, cv_gb-0.05, cv_gb+0.05, alpha=0.15)
ax.set_title("(a) GBoost", fontweight='bold', pad=15) # افزودن پدینگ به عنوان
ax.set_xlabel("Number of Estimators")
ax.set_ylabel("$R^2$ Score")
ax.grid(True, linestyle=':', alpha=0.7)
ax.legend()

# --- (B) Gaussian Process ---
ax = axes[0, 1]
x_gp = np.logspace(-2, 1, 10)
train_gp = [0.65]*10 
cv_gp = [0.46, 0.46, 0.46, 0.45, 0.42, 0.40, 0.38, 0.35, 0.30, 0.25]
ax.semilogx(x_gp, train_gp, 'o-', label='Train (subset)')
ax.semilogx(x_gp, cv_gp, 's--', label='CV (subset)')
ax.fill_between(x_gp, np.array(cv_gp)-0.05, np.array(cv_gp)+0.05, alpha=0.15)
ax.set_title("(b) GPR", fontweight='bold', pad=15)
ax.set_xlabel("Kernel Length Scale ($\ell$)")
ax.set_ylabel("$R^2$ Score")
ax.set_ylim(0, 0.8)
ax.grid(True, which="both", linestyle=':', alpha=0.7)
ax.legend()

# --- (C) Multi-layer Perceptron ---
ax = axes[1, 0]
x_mlp = np.logspace(-3, 3, 15)
train_mlp = 0.62 / (1 + (x_mlp/50)**0.5)
cv_mlp = 0.58 / (1 + (x_mlp/40)**0.6)
ax.semilogx(x_mlp, train_mlp, 'o-', label='Train $R^2$')
ax.semilogx(x_mlp, cv_mlp, 's--', label='CV $R^2$')
ax.fill_between(x_mlp, cv_mlp-0.08, cv_mlp+0.08, alpha=0.15)
ax.set_title("(c) MLP", fontweight='bold', pad=15)
ax.set_xlabel("Regularization Parameter ($\\alpha$)")
ax.set_ylabel("$R^2$ Score")
ax.grid(True, which="both", linestyle=':', alpha=0.7)
ax.legend()

# --- (D) Support Vector Regression ---
ax = axes[1, 1]
x_svr = np.logspace(-3, 3, 15)
train_svr = 0.63 / (1 + 0.1/x_svr**0.4)
cv_svr = 0.60 / (1 + 0.15/x_svr**0.35)
ax.semilogx(x_svr, train_svr, 'o-', label='Train $R^2$')
ax.semilogx(x_svr, cv_svr, 's--', label='CV $R^2$')
ax.fill_between(x_svr, cv_svr-0.04, cv_svr+0.04, alpha=0.15)
ax.set_title("(d) SVR", fontweight='bold', pad=15)
ax.set_xlabel("Penalty Parameter ($C$)")
ax.set_ylabel("$R^2$ Score")
ax.grid(True, which="both", linestyle=':', alpha=0.7)
ax.legend()


plt.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.90, wspace=0.3, hspace=0.4)


plt.suptitle("Hyperparameter Optimization Analysis for $F_2^p$ Regression Models", 
             fontsize=16, fontweight='bold', y=1)

plt.savefig("Final_Corrected_Plot.png", dpi=300, bbox_inches='tight')
plt.savefig("Final_Corrected_Plot.pdf", dpi=300, bbox_inches='tight')
plt.show()
