import csiread
import numpy as np
import matplotlib.pyplot as plt

data = csiread.Atheros("../datasets/data2.dat", nrxnum=2, ntxnum=5, tones=56, if_report=False)
data.read(endian='big')
payload_len = np.bincount(data.payload_len).argmax()
csi = data.csi[(data.payload_len == payload_len) & (data.nc == 2)][:, :, :2, :2]

time = data.timestamp[(data.payload_len == payload_len) & (data.nc == 2)]
t = time/1000000 - time[0]/1000000

subcarrier_index = 15
amplitude = np.abs(csi[:, subcarrier_index])

plt.figure()
plt.plot(t, amplitude[:, 0, 0], linewidth=0.3, label='subcarrier_15_0_0')
plt.plot(t, amplitude[:, 1, 0], linewidth=0.3, label='subcarrier_15_1_0')
plt.plot(t, amplitude[:, 0, 1], linewidth=0.3, label='subcarrier_15_2_0')
plt.legend()

plt.title('csi-amplitude')
plt.xlabel('time(s)')
plt.ylabel('amplitude')
plt.show()

print(csi)