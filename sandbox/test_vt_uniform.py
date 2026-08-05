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

sersegm = segments.SeriesNetwork(
    # segments.DefaultInertance(),
    segments.DefaultViscousLossTF(resistive_only=True),
    segments.DefaultLaminarResistance(),
)
shtsegm = segments.ShuntNetwork(
    # segments.DefaultCompliance(),
    segments.DefaultYieldingWall(),
    segments.DefaultHeatLossGain(),
)
fsegm = segments.DTForwardDelay()
bsegm = segments.DTBackwardDelay()
fsegm_noloss = segments.DTForwardDelay(0)
bsegm_noloss = segments.DTBackwardDelay(0)
jct = junctions.LosslessJunction()
vf_src = endpoints.VFFlowSource()
lips = endpoints.TwoPortAcousticRadiator()
ywall = segments.ShuntNetwork(segments.DefaultYieldingWall())


# discrete-time letalker (without yielding wall)
def iter_letalker():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield fsegm(area, length, **kwargs)
        else:
            yield bsegm(area, length, **kwargs)
        if j is not None:
            yield jct(area, area, **kwargs)
    yield lips(area, **kwargs)


def iter_letalker_ywall():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield ywall(area, length, **kwargs)
            yield fsegm(area, length, **kwargs)
        else:
            yield bsegm(area, length, **kwargs)
            yield ywall(area, length, **kwargs)
        if j is not None:
            yield jct(area, area, **kwargs)
    yield lips(area, **kwargs)


def iter_letalker_shtsegm():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield shtsegm(area, length, **kwargs)
            yield fsegm(area, length, **kwargs)
        else:
            yield bsegm(area, length, **kwargs)
            yield shtsegm(area, length, **kwargs)
        if j is not None:
            yield jct(area, area, **kwargs)
    yield lips(area, **kwargs)


def iter_letalker_ser_segm():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield sersegm(area, length, **kwargs)
            yield fsegm_noloss(area, length, **kwargs)
        else:
            yield bsegm_noloss(area, length, **kwargs)
            yield sersegm(area, length, **kwargs)
        if j is not None:
            yield jct(area, area, **kwargs)
    yield lips(area, **kwargs)


def iter_letalker_segm():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield bsegm(area, length, **kwargs)
            yield sersegm(area, length, **kwargs)
            yield shtsegm(area, length, **kwargs)
        else:
            yield shtsegm(area, length, **kwargs)
            yield sersegm(area, length, **kwargs)
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
domega = omega[1] - omega[0]

fig, axes = plt.subplots(2, 1, sharex=True)
for label, iter in (
    # ("letalker", iter_letalker),
    # ("letalker+yielding_wall", iter_letalker_ywall),
    # ("letalker+yielding_wall+heatloss", iter_letalker_shtsegm),
    # ("letalker+viscous_loss+laminar_loss", iter_letalker_ser_segm),
    ("letalker+all_losses", iter_letalker_segm),
    # ("flaganagn_1pc", iter_flanagan_1pc),
    # ("flaganagn", iter_flanagan),
):
    sys = cascade.chain(*iter(), inplist=[(0, "Ug")], outlist=[(-1, "Prad")])
    resp = sys.frequency_response(omega, squeeze=True)
    axes[0].plot(f, 20 * np.log10(resp.magnitude), label=label)
    axes[1].plot(
        f,
        resp.phase * 180 / np.pi,
        # f[:-1] + domega / 2,
        # -np.angle(resp.complex[1:] / resp.complex[:-1]) / domega * fs,
        label=label,
    )

axes[0].legend()
plt.show()
