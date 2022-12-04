import processing.process
import matplotlib.pyplot as plt

[csi, data] = processing.read.extractCSI('../datasets/air-or-not/second/case_train.dat')
csi = processing.process.extractAm(csi)
csi = processing.process.reshape224x1(csi)

plt.figure()
for i in range(1, 224):
    plt.plot(range(224), csi[i], linewidth=0.3, label='power')
#plt.legend()

plt.title('csi-power')
plt.xlabel('subcarrier index')
plt.ylabel('power')
plt.show()