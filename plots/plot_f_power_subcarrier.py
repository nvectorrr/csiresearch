import processing.process
import matplotlib.pyplot as plt

[csi, data] = processing.read.extractCSI('../datasets/air-or-not/fifth/20MHz/air_train.dat')
csi = processing.process.extractAm(csi)
csi = processing.process.reshape4x56(csi)

plt.figure()
plt.plot(range(56), csi[456][0], linewidth=1.5, label='power')
plt.plot(range(56), csi[456][1], linewidth=1.5, label='power')
plt.plot(range(56), csi[456][2], linewidth=1.5, label='power')
plt.plot(range(56), csi[456][3], linewidth=1.5, label='power')
#plt.legend()

plt.title('csi-power')
plt.xlabel('subcarrier index')
plt.ylabel('power')
plt.grid(linewidth=1.5)
plt.show()

# [csi, data] = processing.read.extractCSI('../datasets/air-or-not/fifth/40MHz/air_train.dat')
# csi = processing.process.extractAm(csi)
# csi = processing.process.reshape4x56(csi)
#
# plt.figure()
# plt.plot(range(114), csi[456][0], linewidth=1.5, label='power') #blue
# plt.plot(range(114), csi[456][1], linewidth=1.5, label='power') #orange
# plt.plot(range(114), csi[456][2], linewidth=1.5, label='power') #green
# plt.plot(range(114), csi[456][3], linewidth=1.5, label='power') #red
# #plt.legend()
#
# plt.title('csi-power')
# plt.xlabel('subcarrier index')
# plt.ylabel('power')
# plt.grid(linewidth=1.5)
# plt.show()