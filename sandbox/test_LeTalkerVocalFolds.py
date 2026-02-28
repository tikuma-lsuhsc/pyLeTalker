import numpy as np
from matplotlib import pyplot as plt

from letalker import sim
from letalker.constants import fs
from letalker.elements import *

N = round(0.1 * fs)

pout, res = sim(N, LeTalkerVocalFolds(), "aa")

# zero-crossing analysis
x = res["vocalfolds"].displacements.min(-1)
rising_edges = np.where((x[:-1] < 0) & (x[1:] > 0))[0]
fo = fs / (rising_edges[-1] - rising_edges[-2])

t = np.arange(N) / fs * 1e3
# plt.plot(t * 1e3, y[:,:3])

fig, ax = plt.subplots(5, 1, sharex=True)
ax[0].axhline(0, ls=":", lw=0.5)
ax[0].plot(t, pout)
ax[0].set_ylabel("Po")
ax[0].set_title(f"F0 = {fo:0.0f} Hz", fontsize=10)

ax[1].axhline(0, ls=":", lw=0.5)
ax[1].plot(t, res["lungs"].plung, label="lungs")
ax[1].plot(t, res["vocalfolds"].subglottal_pressure, label="subglottal")
ax[1].plot(t, res["vocalfolds"].intraglottal_pressure, label="intraglottal")
ax[1].legend()
ax[1].set_ylabel("Ps & Pg")

ug = res["vocalfolds"].glottal_flow
ax[2].plot(t, ug)
ax[2].set_title(f"Max Glottal Flow = {ug.max():0.0f} cm³/s", fontsize=10)
ax[2].set_ylabel("Ug")

ag = res["vocalfolds"].glottal_area
ax[3].plot(t, ag)
ax[3].set_title(f"Max Glottal Area = {ag.max():0.2f} cm²", fontsize=10)
ax[3].set_ylabel("Ag")

ax[4].axhline(0, ls=":", lw=0.5)
ax[4].plot(t, res["vocalfolds"].displacements)
ax[4].legend(["bottom", "top", "body"])
ax[4].set_ylabel("Disp.")
ax[4].set_xlabel("time (ms)")

fig.align_ylabels(ax)

plt.figure()
plt.axvline(0, lw=1, c="k")
dx = res["vocaltract"].dx
a = res["vocaltract"].areas
x = np.arange(len(a) + 1) * dx
plt.step(x, [*a, a[-1]], where="post")
plt.annotate(
    "Vocal tract",
    (x.mean(), 0.95),
    xycoords=("data", "axes fraction"),
    ha="center",
    va="top",
)
dx = res["trachea"].dx
a = res["trachea"].areas
x = np.arange(-len(a), 1) * dx
plt.step(x, [*a, a[-1]], where="post")
plt.annotate(
    "Trachea",
    (x.mean(), 0.95),
    xycoords=("data", "axes fraction"),
    ha="center",
    va="top",
)
plt.xlabel("Distance from glottis (cm)")
plt.ylabel("Area (cm²)")
plt.show()

# save displacement data for visualization test
# import pickle, letalker
# with open(f"{letalker.__path__[0]}/../../sandbox/ltvf_disps.pkl", "wb") as f:
#     pickle.dump(res["vocalfolds"].displacements, f)
