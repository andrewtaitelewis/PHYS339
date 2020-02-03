import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

def function(x, a , b):
    return a*x + b

def lin_fit(x):
    return 0.01967005*x + 0.00429446
def lin_error(x):
    np.sqrt(np.square(5.81624618e-04) + (x**2 *np.square(4.21456817e-06)))
    
d9 = np.loadtxt('Pin9_output.csv', delimiter = ',')
d10 = np.loadtxt('Pin10_output.csv', delimiter = ',')
d10_50 = np.loadtxt('Pin10_50.csv', delimiter = ',')
d10_100 = np.loadtxt('Pin10_100.csv', delimiter = ',')
d10_250 = np.loadtxt('Pin10_250.csv', delimiter = ',')
d10_1 = np.loadtxt('Pin10_120_firstfreq.csv', delimiter = ',')
d10_2 = np.loadtxt('Pin10_120_secfreq.csv', delimiter = ',')
d10_3 = np.loadtxt('Pin10_120_thirdfreq.csv', delimiter = ',')

data = np.loadtxt('A0_calib.csv', delimiter = ',')
A0_means = np.zeros(256)
A0_stds = np.zeros(256)
#importing the raw data is now done. 

for i in range(len(data)):
    A0_means[i] = np.mean(data[i])
    A0_stds[i] = np.std(data[i])
    

linspace = np.linspace(0,255,256)
volt_range = np.zeros(256)
volt_range = lin_fit(linspace)

popt, pcov = curve_fit(function, A0_means, volt_range, sigma = lin_error(volt_range))

perr = np.sqrt(np.diag(pcov))

def A_error(x):
    np.sqrt(np.square(perr[1]) + (x**2)*np.square(perr[0]))
"""
plt.figure()
plt.plot(A0_means, function(A0_means, *popt))

"""



timeScale = np.linspace(0, (2.040*500/25), 500)




plt.figure()
plt.plot(timeScale, function(d10, *popt), '.')
plt.errorbar(timeScale, function(d10, *popt), yerr = A_error(d10),fmt = '.')
plt.ylabel("Voltage(V)")
plt.xlabel("Time (ms)")
plt.title("Output of Pin 10, Duty Cycle 47%, with Time")
plt.legend(["Voltage from Linear Fit", "Error on Fit"])
plt.show()

print(function(d10,*popt))
