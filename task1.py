import numpy as np
import math
import matplotlib.pyplot as plt

Rmax = 11

n = 2501
k = math.sqrt(0.250)
h = Rmax / (n-1)

A = np.zeros((n, n))
s = np.zeros(n)
c = np.zeros(n)
s[n - 1] = math.sin(k * Rmax) / pow(h, 2)
c[n - 1] = math.cos(k * Rmax) / pow(h, 2)
x = np.linspace(0, Rmax, n)

Sin = np.zeros(n)
Cos = np.zeros(n)
V = np.zeros(n)
for i in np.arange(0, n):
    V[i] = -2 * math.exp(-2 * i * h)
    Sin[i] = math.sin(x[i]*k)
    Cos[i] = math.cos(x[i]*k)
for i in np.arange(0, n):
    for j in np.arange(0, n):
        if i == j:
            A[i][j] = 2 / pow(h, 2) + V[i] - pow(k, 2)
        elif (i == j - 1) or (i == j + 1):
            A[i][j] = -1 / pow(h, 2)
phy_c = np.linalg.solve(A, c)
phy_s = np.linalg.solve(A, s)
q = 5
while(phy_s[q]<phy_s[q+1]):
    q = q + 1
max1 = q
while(phy_s[q]>phy_s[q+1]):
    q = q + 1
min1 = q
q = 5
while(math.sin(k*q*h)<math.sin(k*(q+1)*h)):
    q = q + 1
max2 = q
print((max2-max1)*math.pi/(min1 - max2))
tang_delta = (math.sin(k * (Rmax - h)) - phy_s[n - 2]) / (phy_c[n - 2] - math.cos(k * (Rmax - h)))
#print(math.atan(tang_delta))
plt.subplot(221)
plt.plot(x, phy_c)
plt.plot(x, Sin)
#plt.plot(x, Cos)
plt.plot(x, phy_s)
plt.show()
