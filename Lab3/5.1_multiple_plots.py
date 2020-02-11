# -*- coding: utf-8 -*-
"""
Created on Thu Feb 06 16:19:41 2020

@author: ggiamp
"""

import numpy as np
import matplotlib.pyplot as plt
#import random as r

data = np.loadtxt('count2.csv', delimiter = ',')

plt.figure()
plt.title("Mega-histogram of 5 Runs")
plt.xlabel("Decay Events Counted in 1 s")
plt.ylabel("Counts")
n1, b1, p1 = plt.hist(data[0], bins = 15, range = (70,140))
n2, b2, p2 = plt.hist(data[1], bins = 15, range= (70,140))
n3, b3, p3 = plt.hist(data[2], bins = 15, range = (70,140))
n4, b4, p4 = plt.hist(data[3], bins = 15, range  = (70,140))
n5, b5, p5 = plt.hist(data[4], bins = 15, range = (70,140))
plt.show()

centers = np.zeros(15)
for i in range(15):
    centers[i] = 0.5* (b1[i] + b1[i+1])

binMeans = np.zeros(15)
binStds = np.zeros(15)

for i in range(15):
    binMeans[i] = (n1[i] + n2[i] + n3[i] + n4[i] +n5[i]) / 5
    binStds[i] = np.std([n1[i], n2[i], n3[i], n4[i], n5[i]])

plt.figure()
plt.title("Mean Histogram Across 5 runs")
plt.xlabel("Decay Events Recorded / s")
plt.ylabel("Counts")
plt.bar(centers, binMeans, (7/1.5), yerr = binStds)
plt.show()

#ideally, we'd need to plot a gaussian and a possion over this
