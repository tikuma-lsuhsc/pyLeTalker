import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz

from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import cascade, endpoints, junctions, segments

fs = 44100
sample_kws = {"method": "bilinear"}

nsegs = 44

# p.89 Story 95
areas = np.full(nsegs, 4.0)
# areas[:22] = 0.5
# areas[22:] = 3

fsegm = segments.DTForwardDelay()
bsegm = segments.DTBackwardDelay()
jct = junctions.LosslessJunction()

segms = [
    cascade.chain(
        bsegm(area1, length, fs=fs),
        jct(area1, area2, fs=fs),
        fsegm(area2, length, fs=fs),
    )
    for i, (area1, area2) in enumerate(zip(areas[::2], areas[1::2]))
]

sys = segms[0]
for area1, area2, s2 in zip(areas[1::2], areas[2::2], segms[1:]):
    sys = cascade.chain(sys, jct(area1, area2, fs=fs), s2)
    # sys = cascade.chain(sys, s2)

vf_src = endpoints.VFFlowSource()
sys = cascade.cascade(vf_src(areas[0], fs=fs), sys)

lips1 = endpoints.TwoPortAcousticRadiator()
lips2 = endpoints.TwoPortStoryRadiator()  # TwoPortAcousticRadiator()

for lips in (lips1, lips2):
    tf = cascade.cascade(sys, lips(areas[-1], fs=fs, sample_kws=sample_kws)).to_tf()

    f, H = freqz(tf.num[0][0], tf.den[0][0], fs=fs, worN=fs)
    plt.plot(f[1:], 20 * np.log10(np.abs(H[1:])))

plt.xlim(10, 5000)

plt.legend(["1", "2"])
plt.show()
