import librosa
import librosa.feature as lf
import os
import numpy as np
import matplotlib.pyplot as plt
from numpy.core.fromnumeric import mean


mMFCCavg = [
    6163.311513,
-1998.738398,
-205.5361884,
-534.1259459,
-413.9364687,
-184.3113284,
-76.65212511,
-194.1803049,
-130.2207741,
-199.5656133,
-247.4632992,
-198.0631335,
-237.8617047,
-219.4858771,
-258.0808143,
-237.2015379,
-199.3506281,
-211.8046433,
-208.975392,
-207.7573676
]
fMFCCavg = [
    5705.920808,
-1443.281367,
-260.5624223,
-617.5810179,
-197.6205518,
-164.9068898,
-109.783191,
-203.6707464,
-172.2897179,
-201.7449267,
-223.9927003,
-212.7815093,
-227.2048446,
-215.8853454,
-239.530966,
-191.1982349,
-263.879494,
-255.0034166,
-228.818015,
-276.1855251
]

# plt.plot(data=mMFCCavg)
plt.plot(range(0,20),mMFCCavg)
plt.plot(range(0,20),fMFCCavg)
plt.xlabel('Frames')
plt.ylabel('MFCC values')
plt.legend()
plt.show()