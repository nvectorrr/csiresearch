import processing.process
import numpy as np
import matplotlib.pyplot as plt

[csi, data] = processing.read.extractCSI('../datasets/data1.dat')
csi = processing.process.extractAm(csi)
csi = processing.process.reshape4x56(csi)
csi = np.reshape(csi, (csi.shape[0], -1))

payload_len = np.bincount(data.payload_len).argmax()
time = data.timestamp[(data.payload_len == payload_len) & (data.nc == 2)]
time = time/1000000 - time[0]/1000000

subcarrier_index = 15
amplitude = np.abs(csi[:, subcarrier_index])

plt.figure()
plt.plot(time, amplitude, linewidth=0.3, label='subcarrier_15_0_0')
#plt.plot(time, amplitude[:, 1, 0], linewidth=0.3, label='subcarrier_15_1_0')
#plt.plot(time, amplitude[:, 0, 1], linewidth=0.3, label='subcarrier_15_2_0')
plt.legend()

plt.title('csi-amplitude')
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.show()