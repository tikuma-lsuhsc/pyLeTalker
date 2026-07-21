import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import cascade, endpoints, junctions, segments

areas = vocaltract_areas["aa"]

fsegm = segments.DTForwardDelay()
bsegm = segments.DTBackwardDelay()
jct = junctions.LosslessJunction()
shunt = segments.ShuntNetwork(segments.DefaultYieldingWall())

fs = 44100
sample_kws = {"method": "bilinear"}
kwargs = {"fs": fs, "sample_kws": sample_kws}


# def cascade(*systems, **kwargs):
#     return ct.interconnect(
#         systems,
#         connections=[
#             *chain(
#                 *(
#                     [[(i + 1, "F1"), (i, "F2")], [(i, "B2"), (i + 1, "B1")]]
#                     for i in range(len(systems) - 1)
#                 )
#             )
#         ],
#         inplist=[(0, "Ug")],
#         outlist=[(len(systems) - 1, "Prad")],
#         **kwargs,
#     )


lips = endpoints.TwoPortAcousticRadiator()
vf_src = endpoints.VFFlowSource()

from itertools import zip_longest


def iter_letalker(areas):
    yield vf_src(areas[0], **kwargs)
    for i, (area1, area2) in enumerate(zip_longest(areas, areas[1:])):
        if i % 2:
            yield bsegm(area1, length, **kwargs)
        else:
            yield fsegm(area1, length, **kwargs)
        if area2 is not None:
            yield jct(area1, area2, **kwargs)
    yield lips(areas[-1], **kwargs)


sys = cascade.chain(
    *iter_letalker(areas),
    inplist=[(0, "Ug")],
    outlist=[(-1, "Prad")],
)

sys = sys.to_tf()

f, H = freqz(sys.num[0][0], sys.den[0][0], fs=fs, worN=fs)
plt.plot(f[1:], 20 * np.log10(np.abs(H[1:])))
plt.xlim(0, 5000)
plt.show()
