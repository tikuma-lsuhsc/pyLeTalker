import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import cascade, endpoints, junctions, segments

fs = 44100
sample_kws = {"method": "bilinear"}

nsegs = 44
areas = np.full(nsegs, 4.0)

fsegm = segments.DTForwardDelay()
bsegm = segments.DTBackwardDelay()
jct = junctions.LosslessJunction()

segms = [
    cascade.chain(
        bsegm(area1, length, fs=fs, input_id=2 * i + 1, output_id=2 * i + 2),
        # jct(area1, area2, fs=fs),
        fsegm(area2, length, fs=fs, input_id=2 * i + 2, output_id=2 * i + 3),
    )
    for i, (area1, area2) in enumerate(zip(areas[::2], areas[1::2]))
]

sys = segms[0]
for area1, area2, s2 in zip(areas[1::2], areas[2::2], segms[1:]):
    # sys = cascade.chain(sys, jct(area1, area2, fs=fs), s2)
    sys = cascade.chain(sys, s2)


lips = endpoints.TwoPortAcousticRadiator()
# lips = endpoints.TwoPortStoryRadiator()#TwoPortAcousticRadiator()
sys = cascade.cascade(sys, lips(areas[-1], fs=fs, sample_kws=sample_kws))

vf_src = endpoints.VFFlowSource()
sys = cascade.cascade(vf_src(areas[0], fs=fs), sys)

sys = sys.to_tf()

f, H = freqz(sys.num[0][0], sys.den[0][0], fs=fs, worN=fs)
plt.plot(f, 20 * np.log10(np.abs(H)))
plt.xlim(0, 4000)
plt.show()
