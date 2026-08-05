from itertools import zip_longest

import control as ct

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
    segments.DefaultViscousLossTF(resistive_only=False),
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

nsegs = 2


def iter_letalker_ser_segm():
    yield vf_src(area, **kwargs)
    for i, j in zip_longest(range(nsegs), range(1, nsegs)):
        if i % 2:
            yield cascade.combine(
                sersegm(area, length, **kwargs), fsegm_noloss(area, length, **kwargs)
            )
        else:
            yield cascade.combine(
                bsegm_noloss(area, length, **kwargs), sersegm(area, length, **kwargs)
            )
        if j is not None:
            yield jct(area, area, **kwargs)
    yield lips(area, **kwargs)


syschain = [*iter_letalker_ser_segm()]
# sys = cascade.chain(*syschain, inplist=[(0, "Ug")], outlist=[(-1, "Prad")])
# print(sys)
# sys = cascade.chain(
#     *syschain, inplist=[(0, "F1"), (-1, "B2")], outlist=[(0, "B1"), (-1, "F2")]
# )
sys = ct.observable_form(cascade.chain(*syschain[1:3]))
print(sys)
print(syschain[3])
# sys = cascade.chain(*syschain[2:4])
# sys = cascade.chain(*syschain[3:5])
# sys = cascade.chain(*syschain[4:6])
# sys = cascade.chain(
#     *syschain[5:], inplist=[(0, "F1")], outlist=[(-1, "Prad"), (0, "B1")]
# )
