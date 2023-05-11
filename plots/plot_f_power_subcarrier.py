import processing.process
import matplotlib.pyplot as plt

[csi, data] = processing.read.extractCSI('../datasets/air-or-not/second/case_train.dat')
csi = processing.process.extractAm(csi)
csi = processing.process.reshape224x1(csi)

plt.figure()
for i in range(1, 224):
    plt.plot(range(224), csi[i], linewidth=0.3)
#plt.legend()

plt.title('Амплитудные значения CSI для различных пакетов.')
plt.xlabel('Номер поднесущей')
plt.ylabel('Мощность, мВт')
plt.grid()
plt.show()