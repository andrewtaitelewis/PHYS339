#Importing the relevant modules
import numpy as np 
import matplotlib.pyplot as plt 
import os 
from scipy.optimize import curve_fit

def linearFunction(x,a,b):
    """ A simple linear function: y = a*x + b """
    return a*x + b

def main():
    """Our main method for our file"""
    #Importing all the relevant data files
    A1 = np.loadtxt(r"A1_means_stds.csv", delimiter = ',')
    A1 = A1.transpose()


    A2 = np.loadtxt(r"A2_means_stds.csv", delimiter = ',')
    A2 = A2.transpose()
    #Our figure of A1

    #Converting voltage
    linspace = np.linspace(0,255,256)
    slope = 0.01967005

    intercept = 0.00429446
    #Linspace to voltage
    yAxis = linearFunction(linspace,slope,intercept)

    #Converting the error to voltage
    A1ResidualError = A1[1]
    A2ResidualError = A2[1]

    A1[1] = linearFunction(A1[1],slope,intercept)
    A2[1] = linearFunction(A2[1],slope,intercept)

    #Residuals
    A1Residuals = A1[0] - linspace
    A2Residuals = A2[0] - linspace
    
    



    #Our Figure of A1
    plt.figure()
    plt.subplot(1,1,1)

    plt.subplot(2,1,1)
    plt.plot(A1[0],yAxis,'o')
    plt.errorbar(A1[0],yAxis,yerr=A1[1],fmt = '.')
    plt.xlabel('8 Bit Number Read by A1')
    plt.ylabel('Voltage [V]')
    #Our Residuals
    plt.subplot(2,1,2)
    plt.plot(A1[0],A1Residuals,'o')
    plt.axhline(0,color = 'black')
    plt.errorbar(A1[0],A1Residuals,yerr = A1ResidualError ,fmt = '.')

    plt.ylabel('Voltage [V]')
    plt.xlabel('8 Bit Number Read by A1')
    print(A1ResidualError)
    plt.tight_layout()
    plt.savefig('A1')


    #Our figure of A2
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(A2[0],yAxis,'o')
    plt.errorbar(A2[0],yAxis,yerr=A2[1])
    
    plt.ylabel('Voltage [V]')
    #Our Residuals
    plt.subplot(2,1,2)
    plt.plot(A2[0],A2Residuals,'o')
    plt.errorbar(A2[0],A2Residuals,yerr =A2ResidualError, fmt = '.')
    plt.axhline(0,color = 'black')
    plt.xlabel('8 Bit Number Read by A2')
    plt.tight_layout()

    plt.savefig('A2')

    plt.close()
    #Subplots?
    



    return
main()