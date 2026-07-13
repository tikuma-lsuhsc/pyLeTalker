import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import junction, segment

areas = vocaltract_areas["aa"]

fsegm = segment.DTForwardDelay()
bsegm = segment.DTBackwardDelay()
jct = junction.LosslessJunction()

fs = 44100
segms = [
    segment.cascade(
        bsegm(area1, length, fs=fs),
        jct(area1, area2, fs=fs),
        fsegm(area2, length, fs=fs),
    )
    for area1, area2 in zip(areas[::2], areas[1::2])
]

sys = segms[0]
for area1, area2, s2 in zip(areas[1::2], areas[2::2], segms[1:]):
    sys = segment.cascade(sys, jct(area1, area2, fs=fs), s2)


sys = sys.to_tf()

f, H = freqz(sys.num[0][0], sys.den[0][0], fs=fs)
plt.plot(f, 20 * np.log10(np.abs(H)))
plt.show()
