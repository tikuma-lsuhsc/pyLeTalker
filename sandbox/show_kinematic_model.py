from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import axes3d
import numpy as np

from time import sleep

from letalker import KinematicVocalFolds, SineGenerator, ModulatedSineGenerator
from letalker.constants import fs, male_vf_params, rho_air, c, vocaltract_areas, PL

N = 1000

t = np.arange(N) / fs

fo = 101

m_freq = 1 / 2
extent = 0.2

xiref = SineGenerator(fo)
# xiref = ModulatedSineGenerator(fo, m_freq, am_extent=extent)
# xiref = ModulatedSineGenerator(fo, m_freq, fm_extent=extent)
# xiref = ModulatedSineGenerator(
#     [fo, fo * 3 / 2], m_freq, am_extent=extent, fm_extent=extent
# )

Ae = vocaltract_areas["aa"][0]
As = vocaltract_areas["trach"][-1]

vf = KinematicVocalFolds(
    xiref, As=As, Ae=Ae  # , **male_vf_params, Qa=0.3, Qs=3, Qb=1, Qp=0.2
)

fig = plt.figure()
ax = fig.add_subplot(projection="3d")

# ax.view_init(22.5, -70) # good perspective view
ax.view_init(90, -90)  # top view
ax.set_proj_type("ortho")
ax.set_aspect("equal")

vf.draw3d(0, axes=ax, draw_box=True)
with plt.ion():
    plt.pause(0.001)
    for i in range(1, N):
        vf.draw3d(i, axes=ax, update=True)
        plt.pause(0.001)
