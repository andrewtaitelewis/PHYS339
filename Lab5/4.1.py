#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 14:35:39 2020

@author: gabriel_giampa
"""

import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns; sns.set()
sns.set_palette("bright")
sns.set_style("white")

"""
bright=["#023EFF", "#FF7C00", "#1AC938", "#E8000B", "#8B2BE2",
            "#9F4800", "#F14CC1", "#A3A3A3", "#FFC400", "#00D7FF"]


 - blue: "#023EFF"
 - Orange: "#FF7C00"
 - Green: "#1AC938"
 - Red: "#E8000B"
 - Purple: "#8B2BE2"
 - Brown: "#9F4800"
 - Pink: "#F14CC1"
 - Gray: "#A3A3A3"
 - Yellow: "#FFC400"
 - Cyan: "#00D7FF"
"""

sns.palplot(sns.color_palette("bright"))

data = np.loadtxt('4.1_tset-hitting.csv', delimiter = ',')
data = np.transpose(data)

plt.figure()
plt.title("Temperature of Bar Controlled By On-Off Algorithm")
plt.ylabel("Temperature (K)", fontsize = 14)
plt.errorbar(data[4], data[3], yerr = data[2], color = 'red', label = "Error", fmt = 'none')
plt.xlabel("Time Step (~0.1 s)", fontsize = 14)
plt.plot(data[4],data[3], label = "Temperature", color = 'Purple')
plt.axhline(350, ls = '--',label = "Setpoint", color = 'black')
plt.xlim(100, len(data[3]) - 1)
plt.ylim(340,355)
plt.legend()
plt.show()

data2 = np.transpose(np.loadtxt("4.1_b5-2-1.csv", delimiter = ','))

plt.figure()
plt.title("Temperature of Bar Controlled By On-Off Algorithm, Version 2")
plt.plot(data2[4],data2[3], label = "Temperature", color = 'Purple')
plt.errorbar(data2[4], data2[3], yerr = data2[2], fmt = 'none', color = 'red', label = 'Error')
plt.axhline(350, ls = '--', label = 'Setpoint', color = 'black')
plt.ylabel("Temperature(K)", fontsize = 14)
plt.xlabel("Time step (~0.1s)", fontsize = 14)
#plt.axhline(352.5, color = 'green', lw = 2)
#plt.axhline(347.5, color = 'green', lw = 2)
plt.xlim(200, len(data2[3]) - 1)
plt.ylim(330,355)
plt.legend()
plt.show()