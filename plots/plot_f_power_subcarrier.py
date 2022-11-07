import processing.process
import matplotlib.pyplot as plt

[csi, data] = processing.read.extractCSI('../datasets/data1.dat')
csi = processing.process.extractAm(csi)
csi = processing.process.reshape224x1(csi)

plt.figure()
plt.plot(range(224), csi[1], linewidth=0.3, label='power')
plt.legend()

plt.title('csi-power')
plt.xlabel('subcarrier index')
plt.ylabel('power')
plt.show()