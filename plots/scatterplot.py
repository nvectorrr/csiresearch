import matplotlib.pyplot as plt

accr_raw_20 = [91.2, 99.5, 95.4]
time_raw_20 = [24.5, 3.1, 0.8]

accr_pca_20 = [99.8, 99.2, 95.3]
time_pca_20 = [14.3, 2.4, 1.3]

accr_red_20 = [93.8, 99.3, 97.9]
time_red_20 = [39.5, 37.1, 36.1]

accr_com_20 = [99.8, 99.1, 95.6]
time_com_20 = [19.5, 19.7, 15.9]

accr_raw_40 = [89.5, 96.7, 97.4]
time_raw_40 = [23.1, 12.5, 1.6]

accr_pca_40 = [90.6, 93.7, 97.4]
time_pca_40 = [10.8, 3.7, 2.5]

accr_red_40 = [90.5, 94.8, 96.8]
time_red_40 = [72.3, 69.4, 70.7]

accr_com_40 = [91.1, 94.5, 97.7]
time_com_40 = [22.8, 21.1, 22.6]

plt.scatter(accr_raw_20, time_raw_20, c="blue")
plt.scatter(accr_pca_20, time_pca_20, c="red")
plt.scatter(accr_red_20, time_red_20, c="orange")
plt.scatter(accr_com_20, time_com_20, c="green")

plt.grid()
plt.show()

plt.figure()

plt.scatter(accr_raw_40, time_raw_40, c="blue")
plt.scatter(accr_pca_40, time_pca_40, c="red")
plt.scatter(accr_red_40, time_red_40, c="orange")
plt.scatter(accr_com_40, time_com_40, c="green")

plt.grid()
plt.show()

# --------------------------------------------
# accr_rfc_20 = [91.2, 99.8, 93.8, 99.8]
# time_rfc_20 = [24.5, 14.3, 39.5, 19.5]
#
# accr_svm_20 = [99.5, 99.2, 99.3, 99.1]
# time_svm_20 = [3.1, 2.4, 37.1, 19.7]
#
# accr_knn_20 = [95.4, 95.3, 97.9, 95.6]
# time_knn_20 = [0.8, 1.3, 36.1, 15.9]
#
# accr_rfc_40 = [89.5, 90.6, 90.5, 91.1]
# time_rfc_40 = [23.1, 10.8, 72.3, 22.8]