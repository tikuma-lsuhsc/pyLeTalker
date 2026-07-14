import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import junctions, segments, cascade, endpoints


def test_junction():
    areas = vocaltract_areas["aa"]

    jct = junctions.LosslessJunction()

    fs = 44100
    area1, area2 = areas[:2]

    r1 = (area1 - area2) / (area1 + area2)
    r2 = -r1
    D = np.array([[1 + r1, r2], [r1, 1 + r2]])

    sys1 = jct(area1, area2, fs=fs, _force_mimo=False)
    sys2 = jct(area1, area2, fs=fs, _force_mimo=True)

    assert np.allclose(sys1.D, D)
    assert np.allclose(sys2.D, D)


def _test_chain():
    areas = vocaltract_areas["aa"]

    fsegm = segments.DTForwardDelay()
    bsegm = segments.DTBackwardDelay()
    jct = junctions.LosslessJunction()

    fs = 44100
    area1, area2 = areas[:2]

    assert jct(area1, area2, fs=fs, _force_mimo=False) == jct(
        area1, area2, fs=fs, _force_mimo=True
    )

    sys = cascade.cascade(bsegm(area1, length, fs=fs), jct(area1, area2, fs=fs))
    sys = cascade.cascade(sys, fsegm(area2, length, fs=fs))
    print(sys)


def _test_vt_components_letalker():
    areas = vocaltract_areas["aa"]

    fsegm = segments.DTForwardDelay()
    bsegm = segments.DTBackwardDelay()
    jct = junctions.LosslessJunction()

    fs = 44100
    segms = [
        cascade.chain(
            bsegm(area1, length, fs=fs),
            jct(area1, area2, fs=fs),
            fsegm(area2, length, fs=fs),
        )
        for area1, area2 in zip(areas[::2], areas[1::2])
    ]

    sys = segms[0]
    for area1, area2, s2 in zip(areas[1::2], areas[2::2], segms[1:]):
        sys = cascade.chain(sys, jct(area1, area2, fs=fs), s2)

    lips = endpoints.TwoPortFlanaganRadiator()
    sys = cascade.cascade(sys,lips(areas[-1],fs=fs,sample_kws={'method':'bilinear'}))

    sys = sys.to_tf()

    f, H = freqz(sys.num[0][0], sys.den[0][0], fs=fs)
    plt.plot(f, 20 * np.log10(np.abs(H)))
    plt.show()


if __name__ == "__main__":
    # _test_chain()
    _test_vt_components_letalker()
