from matplotlib import pyplot as plt
import numpy as np
from math import ceil, log2
from scipy.signal import get_window

from letalker.constants import fs, male_vf_params, female_vf_params, rho_tis
from letalker.elements.KinematicVocalFolds import fo_to_L

print(male_vf_params)

fo = np.linspace(90, 450, 101)
L0 = male_vf_params["L0"]
A, B = male_vf_params["fo2L"]
a, b, eps0 = 11.66, -0.1384, 7.026

L0f = female_vf_params["L0"]
Af, Bf = female_vf_params["fo2L"]


L = fo_to_L(fo, L0, A, B, rho_tis)

eps_to_mu_chan = lambda eps, A, B: A * eps * np.exp(B * eps)
eps_to_mu_riede = lambda eps, A, B: A * np.exp(B * eps)

eps = np.linspace(0, 0.5, 101)
plt.plot(eps * 100, eps_to_mu_chan(eps, A, B), label="Male (Chan 2007)")
plt.plot(eps * 100, eps_to_mu_riede(eps, A, B), "--", c="C0", label="Male (Titze 2016)")
plt.plot(eps * 100, eps_to_mu_chan(eps, Af, Bf), c="C1", label="Chan 2007")
plt.plot(eps * 100, eps_to_mu_riede(eps, Af, Bf), "--", c="C1", label="Titze 2016")
plt.legend()
plt.xlabel('strain $\epsilon$ (%)')
plt.ylabel('stress (kPa)')
plt.show()

print(A, B)

eps = (L - L0) / L0
plt.plot(fo, L)
plt.plot(1 / (2 * L) * np.sqrt(A / rho_tis * eps * np.exp(B * eps)), L, "--")
plt.plot(1 / (2 * L) * np.sqrt(A / rho_tis * np.exp(B * eps)), L, "--")

fof = np.linspace(120, 800, 101)
Lf = fo_to_L(fof, L0f, *female_vf_params["fo2L"], rho_tis)
epsf = (Lf - L0) / L0
plt.plot(fof, Lf)
plt.plot(1 / (2 * Lf) * np.sqrt(A / rho_tis * epsf * np.exp(B * epsf)), Lf, "--")

plt.show()
