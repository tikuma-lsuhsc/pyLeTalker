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

segms = [
    cascade.chain(
        bsegm(area1, length, **kwargs),
        shunt(area1, length, **kwargs, sample_last=True),
        jct(area1, area2, **kwargs),
        fsegm(area2, length, **kwargs),
        shunt(area2, length, **kwargs, sample_last=True),
    )
    for i, (area1, area2) in enumerate(zip(areas[::2], areas[1::2]))
]

sys = segms[0]
for area1, area2, s2 in zip(areas[1::2], areas[2::2], segms[1:]):
    sys = cascade.chain(sys, jct(area1, area2, **kwargs), s2)


lips = endpoints.TwoPortAcousticRadiator()
# lips = endpoints.TwoPortStoryRadiator()#TwoPortAcousticRadiator()
sys = cascade.cascade(sys, lips(areas[-1], **kwargs))

vf_src = endpoints.VFFlowSource()
sys = cascade.cascade(vf_src(areas[0], **kwargs), sys)

sys = sys.to_tf()

f, H = freqz(sys.num[0][0], sys.den[0][0], fs=fs, worN=fs)
plt.plot(f[1:], 20 * np.log10(np.abs(H[1:])))
plt.xlim(0, 5000)
plt.show()
