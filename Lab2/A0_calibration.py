import numpy as np
import matplotlib.pyplot as plt

#This is the linear fit, which relates an 8-bit input on pin 9 to a voltage
def lin_fit(x):
    return 0.01967005*x + 0.00429446 

#This is the associated error with the linear fit
def lin_error(x):
    np.sqrt(np.square(5.81624618e-04) + (x**2 *np.square(4.21456817e-06)))

data = np.loadtxt('A0_calib.csv', delimiter = ',')
A0_means = np.zeros(256)
A0_stds = np.zeros(256)
#importing the raw data is now done. 


for i in range(len(data)):
    A0_means[i] = np.mean(data[i])
    A0_stds[i] = np.std(data[i])
#We have an array of means and an array of stds

linspace = np.linspace(0,255,256)
volt_range = np.zeros(256)
volt_range = lin_fit(linspace)
#Now we have the range of voltages pin 9 supplied in the experiment

volt_error = np.zeros(256)
volt_error = lin_error(linspace*50)
#and the associated uncertainty on every voltage

fig, (ax1,ax2) = plt.subplots(2,1)

ax1.plot(A0_means, volt_range, 'o')
ax1.errorbar(A0_means, volt_range, xerr = A0_stds, fmt = '.')
ax1.errorbar(A0_means, volt_range, yerr = volt_error, fmt = '.')
ax1.set_ylabel("A0 Voltage (V)")
ax1.set_xlabel("Mean Reading at A0")
ax1.set_facecolor((0.97,0.97,0.97))
ax1.set_title("Voltage on A0 as its Mean Readings Change")
ax1.legend(['Mean A0 reading', "Std of Mean Reading", 'Error In Voltage (V)'])

ax2.plot(A0_means, (A0_means - linspace), 'o')
ax2.set_facecolor((0.97,0.97,0.97))
ax2.set_ylabel("A0 readings - Pin 9's output")
ax2.set_xlabel("A0 mean Reading")
ax2.set_label("Mean Reading at A0")
ax2.errorbar(A0_means, (A0_means - linspace), xerr = A0_stds, fmt = '.')
ax2.errorbar(A0_means, (A0_means - linspace), yerr = A0_stds, fmt = '.')
ax2.legend(['Residuals', 'Error in Mean Readings', 'Error in Difference'], prop ={'size': 9})

fig.tight_layout()
fig.align_ylabels()
