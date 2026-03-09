import numpy as np
import matplotlib.pyplot as plt



ekv_1 = np.loadtxt('ekvator_1.txt')
ekv_10 = np.loadtxt('ekvator_10.txt')
ekv_100 = np.loadtxt('ekvator_100.txt')


pol_1 = np.loadtxt('pol_1.txt')
pol_10 = np.loadtxt('pol_10.txt')
pol_100 = np.loadtxt('pol_100.txt')

plt.plot(ekv_1)
plt.plot(ekv_10)
plt.plot(ekv_100)


plt.figure()
plt.plot(pol_1)
plt.plot(pol_10)
plt.plot(pol_100)


ekv_1000 = np.loadtxt('ekvator_1000.txt')
ekv_10000 = np.loadtxt('ekvator_10000.txt')
plt.plot(ekv_1000, '.')
plt.plot(ekv_10000)