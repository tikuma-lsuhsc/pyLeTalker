import control as ct
import numpy as np
from matplotlib import pyplot as plt
from scipy.signal import freqz

from letalker.constants import vocaltract_areas
from letalker.constants import vocaltract_resolution as length
from letalker.vt_components import cascade, endpoints, junctions, segments

sample_kws = {"method": "bilinear"}

lips = endpoints.TwoPortAcousticRadiator()

area = vocaltract_areas["aa"][-1]
fs = 44100
sys = lips(area, fs=fs, sample_kws=sample_kws).to_tf()
ss = endpoints.FlanaganRadiationLoad()(area, fs=fs, sample_kws=sample_kws)
ss1 = endpoints.TwoPortStoryRadiator()(area, fs=fs, sample_kws=sample_kws)

print(sys.poles())
print(ss.poles())
print(ss1.poles())

[f,H] = freqz(ss1.num[0][0],ss1.den[0][0],1024,fs=fs)
plt.plot(f,20*np.log10(np.abs(H)))

[f,H] = freqz(sys.num[0][0],sys.den[0][0],1024,fs=fs)
plt.plot(f,20*np.log10(np.abs(H)))


ct.bode_plot(ss)

plt.show()
