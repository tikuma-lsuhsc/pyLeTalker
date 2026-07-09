import control as ct
from matplotlib import pyplot as plt

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import segment

areas = vocaltract_areas["aa"]

heat_loss = segment.DefaultHeatLossGain()
yielding_wall = segment.DefaultYieldingWall()
laminar_resistance = segment.DefaultLaminarResistance()
viscous_loss = segment.DefaultViscousLossTF()

shunt = segment.ShuntNetwork(yielding_wall, heat_loss)
series = segment.SeriesNetwork(viscous_loss, laminar_resistance)
t_seg = segment.TSegment(series, shunt)
pi_seg = segment.PiSegment(series, shunt)

# H1, H2 = yielding_wall(areas[0], length), heat_loss(areas[0], length)
# fs = 44100
# sys = shunt(areas[0], length, fs=fs, sample_kws={"method": "bilinear"})
# print(sys.ninputs)
# x = np.random.randn(2, 10000)
# resp = ct.forced_response(sys, inputs=x)

# p1 = x[0] + resp.outputs[1]
# p2 = x[1] + resp.outputs[0]
# plt.plot(p1)
# plt.plot(p2)
# plt.show()
sys1 = shunt(areas[0], length)
sys2 = series(areas[0], length)
sys3 = segment.cascade(sys1, sys2)
# sys3 = t_seg(areas[0], length)
# print(sys3)
ct.bode_plot([sys1, sys2, sys3], dB=True, Hz=True)
plt.show()
exit()
# response = ct.frequency_response([sys1, sys2])
# ct.bode_plot([sys1, sys2], overlay_inputs=True, overlay_outputs=True)
ct.bode_plot([H1, H2, ct.parallel(H1, H2)], dB=True, Hz=True)

ct.bode_plot([sys1], dB=True, Hz=True)

# ct.bode_plot(yielding_wall(areas[0], length), dB=True, Hz=True)
# ct.bode_plot(viscous_loss(areas[0], length))
plt.show()
