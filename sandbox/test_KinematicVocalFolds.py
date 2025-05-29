import numpy as np
from matplotlib import pyplot as plt

from letalker import sim, use_numba
from letalker.elements import *
from letalker.constants import fs, nu

use_numba(False)
N = round(0.1 * fs)

print(nu)

pout, res = sim(N, KinematicVocalFolds(100), "aa")

# zero-crossing analysis
fo = res["vocalfolds"].fo

t = np.arange(N) / fs * 1e3
# plt.plot(t * 1e3, y[:,:3])

fig, ax = plt.subplots(5, 1, sharex=True)
ax[0].axhline(0, ls=":", lw=0.5)
ax[0].plot(t, pout)
ax[0].set_ylabel("Po")
ax[0].set_title(f"F0 = {fo:0.0f} Hz", fontsize=10)

ax[1].axhline(0, ls=":", lw=0.5)
ax[1].plot(t, res["lungs"].plung, label="lungs")
ax[1].plot(t, res["vocalfolds"].psg, label="subglottal")
ax[1].plot(t, res["vocalfolds"].peplx, label="epiglottal")
ax[1].legend()
ax[1].set_ylabel("Ps & Pg")

ug = res["vocalfolds"].ug
ax[2].plot(t, ug)
ax[2].set_title(f"Max Glottal Flow = {ug.max():0.0f} cm³/s", fontsize=10)
ax[2].set_ylabel("Ug")

ag = res["vocalfolds"].glottal_area
ax[3].plot(t, ag)
ax[3].set_title(
    f"Max Glottal Area = {ag.max():0.2f} cm²; Glottal Length = {res['vocalfolds'].length}",
    fontsize=10,
)
ax[3].set_ylabel("Ag")

ax[4].axhline(0, ls=":", lw=0.5)
ax[4].plot(t, res["vocalfolds"].displacements)
ax[4].set_ylabel("Disp.")
ax[4].set_xlabel("time (ms)")

fig.align_ylabels(ax)


# save displacement data for visualization test
# import pickle, letalker
# with open(f"{letalker.__path__[0]}/../../sandbox/ltvf_disps.pkl", "wb") as f:
#     pickle.dump(res["vocalfolds"].displacements, f)

plt.figure()
plt.plot(t, res["vocalfolds"].reynolds_number)

plt.show()
