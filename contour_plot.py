import matplotlib.pyplot as pyplot
import numpy as numpy
import seaborn as sns
import scipy as sp 

sns.set_theme()

c_0 = 1.3
M_0 = 5
N_0 = 64

def pr(a, i, N=N_0, c=c_0, M=M_0):
	c_a = np.sin(np.pi*a/2.)*sp.special.gamma(a)/np.pi
    inv = 1/(c**(a*(i-1)))
    c_M = c**(M-i)
    one_minus_c_a = 1 - c**(-a)
    left = (N*c_a*inv*one_minus_c_a/c_M)**(c_M)
    right = (1 - c_a*inv*one_minus_c_a/(1 - c_M/N))**(N - c_M)
    return left*right

a = np.linspace(0, 2, 50)
i = np.linspace(0, M_0, 50)
A, I = np.meshgrid(a, i)
P = pr(A, I)


plt.contourf(A, I, P, cmap='mako');
plt.colorbar()
plt.xlabel(r"$\alpha$")
plt.ylabel(r"$i$")
plt.xticks(np.arange(0, 2, .5))
plt.yticks(np.arange(0, M_0+0.5, .5))
plt.yticks(fontname = "Times New Roman")  
plt.xticks(fontname = "Times New Roman")  
plt.savefig('contour_plot.svg', format='svg', dpi=1200)