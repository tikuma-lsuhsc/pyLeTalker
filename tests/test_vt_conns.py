import control as ct
import numpy as np
from matplotlib import pyplot as plt

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import junctions, segments

def _test_vt_components():
    areas = vocaltract_areas["aa"]

    heat_loss = segments.DefaultHeatLossGain()
    yielding_wall = segments.DefaultYieldingWall()
    laminar_resistance = segments.DefaultLaminarResistance()
    viscous_loss = segments.DefaultViscousLossTF()

    shunt = segments.ShuntNetwork(yielding_wall, heat_loss)
    series = segments.SeriesNetwork(viscous_loss, laminar_resistance)
    # # shunt = segments.ShuntNetwork()
    # # series = segments.SeriesNetwork()
    # t_seg = segments.TSegment(series, shunt)
    # pi_seg = segments.PiSegment(series, shunt)

    # H1, H2 = yielding_wall(areas[0], length), heat_loss(areas[0], length)
    fs = 44100
    # sys = shunt(areas[0], length, fs=fs, sample_kws={"method": "bilinear"})
    # print(sys.ninputs)
    # x = np.random.randn(2, 10000)
    # resp = ct.forced_response(sys, inputs=x)

    # p1 = x[0] + resp.outputs[1]
    # p2 = x[1] + resp.outputs[0]
    # plt.plot(p1)
    # plt.plot(p2)
    # plt.show()
    segm = segments.TSegment()
    jct = junctions.LosslessJunction()

    # sys2 = segments.PiSegment()(areas[0], length)
    sys3 = segments.cascade(
        segments.ShuntNetwork()(areas[0], length), segments.SeriesNetwork()(areas[0], length)
    )

    s1 = segm(areas[0], length)
    j = jct(areas[0], areas[1])
    s2 = segm(areas[1], length)

    sys1 = segments.cascade(s1, j)
    sys2 = segments.cascade(j, s2)
    sample_kws = {"method": "bilinear"}
    sys = segm(areas[0], length, fs=fs, sample_kws=sample_kws)
    for area, prev_area in zip(areas[1:], areas[:-1]):
        sys = segments.cascade(sys, jct(prev_area, area, fs=fs, sample_kws=sample_kws))
        sys = segments.cascade(sys, segm(area, length, fs=fs, sample_kws=sample_kws))

    # print(sys)
    # ct.bode_plot(
    #     sys,
    #     dB=True,
    #     Hz=True,
    #     omega_limits=(1, np.pi * 10000),
    # )


    # sys3 = t_seg(areas[0], length)
    # print(sys3)
    # ct.bode_plot([sys1, sys2, sys3], dB=True, Hz=True, omega_limits=(1, np.pi * 10000))
    # plt.show()
