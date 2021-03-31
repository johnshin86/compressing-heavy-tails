from scipy.stats import levy_stable
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme()

alpha, beta = 1.9, 0.0
mean, var, skew, kurt = levy_stable.stats(alpha, beta, moments='mvsk')

x = np.linspace(levy_stable.ppf(0.01, alpha, beta), levy_stable.ppf(0.99, alpha, beta), 100)

fig, ax = plt.subplots(1, 1)
ax.plot(x, levy_stable.pdf(x, alpha, beta), 'r-', lw=3, alpha=0.8, label = r'$\alpha$ = 1.9')
ax.plot(x, levy_stable.pdf(x, alpha=1.5, beta=0), 'b-', lw=3, alpha=0.8, label = r'$\alpha$ = 1.5')
ax.plot(x, levy_stable.pdf(x, alpha=1., beta=0), 'g-', lw=3, alpha=0.8, label = r'$\alpha$ = 1.')
ax.plot(x, levy_stable.pdf(x, alpha=.5, beta=0), 'y-', lw=3, alpha=0.8, label = r'$\alpha$ = 0.5')
plt.legend()
plt.xlabel(r"$x$")
plt.ylabel(r"$p(x)$")
plt.show()