from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np
import pickle, letalker
from letalker.elements import LeTalkerVocalFolds
from letalker.constants import fs

vf = LeTalkerVocalFolds()

with open(f"{letalker.__path__[0]}/../../sandbox/ltvf_disps.pkl", "rb") as f:
    x = pickle.load(f)

fig = plt.figure()
axes = fig.add_subplot(projection="3d")

axes.view_init(22.5, -70)  # good perspective view
# axes.view_init(90, -90)  # top view
# axes.view_init(0, -90)  # top view

axes.set_proj_type("ortho")
axes.set_aspect("equal")

N = x.shape[0]
t = np.arange(N) / fs
vf.set_geometry(N)

axes.set_xlabel("x")
axes.set_ylabel("y")
axes.set_zlabel("z")

vf.draw3d(0, x[0], axes=axes)
with plt.ion():
    plt.pause(0.001)
    for i, xi in enumerate(x[1:]):
        vf.draw3d(i + 1, xi, axes=axes, update=True)
        plt.pause(0.001)
