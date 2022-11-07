import processing.process
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from utils import calib

[csi, data] = processing.read.extractCSI('../datasets/data1.dat')
s_index = [-56, -54, -52, -50, -48, -46, -44, -42, -40, -38, -36, -34, -32, -30, -28, -26, -24, -22, -20, -18, -16, -14, -12, -10, -8, -6, -4, -2, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40, 42, 44, 46, 48, 50, 52, 54, 56]
phase = np.unwrap(np.angle(csi), axis=1)
phase = calib(phase, s_index)

plt.figure(4)
plt.plot(s_index, phase[:100, :, 0, 0].T, 'r-', linewidth=0.3)
plt.plot(s_index, phase[:100, :, 1, 0].T, 'g-', linewidth=0.3)
#plt.plot(s_index, phase[:100, :, 2, 0].T, 'y-', linewidth=0.3)

patch_1 = mpatches.Patch(color='red', label=':100_r0t0')
patch_2 = mpatches.Patch(color='green', label=':100_r1t0')
patch_3 = mpatches.Patch(color='yellow', label=':100_r2t0')
plt.legend(handles=[patch_1, patch_2, patch_3])

plt.title('csi-phase')
plt.xlabel('subcarriers')
plt.ylabel('phase')
plt.show()