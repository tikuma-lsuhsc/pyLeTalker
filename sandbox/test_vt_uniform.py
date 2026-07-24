from itertools import zip_longest

import numpy as np
from matplotlib import pyplot as plt

from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import cascade, endpoints, junctions, segments

fs = 44100
sample_kws = {"method": "bilinear"}
kwargs = {"fs": fs, "sample_kws": sample_kws, "sample_last": True}

nsegs = 44

# p.89 Story 95
area = 4.0
# areas[:22] = 0.5
# areas[22:] = 3

segm = segments.SeriesNetwork(
    segments.DefaultLosslessPropagationTF(),
    segments.DefaultViscousLossTF(resistive_only=True),
)
fsegm = segments.DTForwardDelay()
bsegm = segments.DTBackwardDelay()
jct = junctions.LosslessJunction()
vf_src = endpoints.VFFlowSource()
lips = endpoints.TwoPortAcousticRadiator()


# discrete-time letalker (without yielding wall)
def iter_letalker():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield bsegm(area, length, **kwargs)
        else:
            yield fsegm(area, length, **kwargs)
        if j is not None:
            yield jct(area, area, **kwargs)
    yield lips(area, **kwargs)


# continuous-time Flanagan equivalent circuit (single segment)
def iter_flanagan_1pc():
    yield vf_src(area, **kwargs)
    yield segm(area, length * nsegs, **kwargs)
    yield lips(area, **kwargs)


# continuous-time Flanagan equivalent circuit (multi-segments)
def iter_flanagan():
    yield vf_src(area)
    for i in range(nsegs):
        yield segm(area, length)
    yield lips(area)


f = np.arange(5000)[10:]
omega = 2 * np.pi * f

for label, iter in (
    ("flaganagn_1pc", iter_flanagan_1pc),
    # ("letalker", iter_letalker),
    # ("flaganagn", iter_flanagan),
):
    sys = cascade.chain(*iter(), inplist=[(0, "Ug")], outlist=[(-1, "Prad")])
    resp = sys.frequency_response(omega, squeeze=True)
    plt.plot(f, 20 * np.log10(resp.magnitude), label=label)

plt.legend()
plt.show()
