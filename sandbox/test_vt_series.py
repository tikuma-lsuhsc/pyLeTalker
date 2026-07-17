import control as ct

from letalker.constants import vocaltract_areas
from letalker.vt_components import junctions, segments

areas = vocaltract_areas["aa"]
fs = 44100

fsegm = segments.DTForwardDelay()
bsegm = segments.DTBackwardDelay()
jct = junctions.LosslessJunction()

sample_kws = {"method": "bilinear"}

L = 2
R = 4

Z = ct.tf([L, R], [1.0])
print((1 / Z).to_ss())
