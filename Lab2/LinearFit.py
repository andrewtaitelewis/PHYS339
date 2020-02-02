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
    print(A1[0])

    #Residuals
    A1Residuals = A1[0] - linspace
    A2Residuals = A2[0] - linspace
    
    



    #Our Figure of A1
    def figure1():
        '''Plots Figure 1 '''
        fig,(ax1,ax2) = plt.subplots(2,1)
        #First subplot,i.e plot

        ax1.plot(A1[0],yAxis,'o')
        ax1.errorbar(A1[0],yAxis,yerr=A1[1],fmt = '.')
        ax1.set_ylabel('Voltage [V]')
        ax1.set_xlabel('8 Bit Number Read by A1 ')
        ax1.set_facecolor((.97,.97,.97))
        ax1.legend(['Number Read','Error Bars'])
        #Residuals Subplot

        ax2.plot(A1[0],A1Residuals,'o', label ='Number Read by A1 ' )
        ax2.axhline(0,color = 'black')
        ax2.errorbar(A1[0],A1Residuals,yerr = A1ResidualError ,fmt = '.', label = 'Error Bars')
        ax2.set_ylabel('Number Read by A1')
        ax2.set_xlabel('8 Bit Number Read by A1')
        ax2.set_facecolor((.97,.97,.97))
        ax2.legend()
        #Figure settings
        ax1.set_title('8 Bit Number Read Given a Voltage')
        
        fig.tight_layout()
        fig.align_ylabels()

        plt.savefig('A1')
        return
        #Figure for A2
    #Our Figure of A2
    def figure2():
        '''Plots Figure 2 '''
        fig,(ax1,ax2) = plt.subplots(2,1)
        #First subplot,i.e plot

        ax1.plot(A2[0],yAxis,'o', label = 'Number Read by A2')
        ax1.errorbar(A2[0],yAxis,yerr=A2[1],fmt = '.', label = 'Error Bars')
        ax1.set_ylabel('Voltage [V]')
        ax1.set_xlabel('8 Bit Number Read by A2 ')
        ax1.set_facecolor((.97,.97,.97))
        ax1.legend()
        #Residuals Subplot

        ax2.plot(A2[0],A2Residuals,'o', label = 'Number Read')
        ax2.axhline(0,color = 'black')
        ax2.errorbar(A2[0],A2Residuals,yerr = A2ResidualError ,fmt = '.', label = 'Error Bars')
        ax2.set_ylabel('Number Read')
        ax2.set_xlabel('8 Bit Number Read by A2')
        ax2.legend()
        ax2.set_facecolor((.97,.97,.97))
        #Figure settings
        ax1.set_title('8 Bit Number Read Given a Voltage')
        
        fig.tight_layout()
        fig.align_ylabels()

        plt.savefig('A2')
        return
    #Linear fit and residuals
    def linearCalibration():
        #Data manipulation

        bits= np.array([0,5,10,20,30,40,50,65,75,90,100,115,130,150,175,200,215,230,245,255])
        volts = np.array([0.004,0.102,0.200,0.397,0.594,0.791,0.988,1.283,1.480,1.775,1.972,2.267,2.562,2.955,3.45,3.94,4.23,4.53,4.82,5.02])
        fit = np.polyfit(bits,volts,1)

        voltserr = [0.001]*20
        def residualNumbers(x,y):
            residualValues = x-y
            return residualValues

        def slope(x,y):
            a = (y[-1] - y[0])/(x[-1] - x[0])
            return a
        def func(x,a,b):
            return a*x + b
        y = func(bits,slope(bits,volts),0.004)

        popt, pcov = curve_fit(func, bits, volts)

        #Now for error bars
        error = np.array([0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.001,0.01,0.01,0.01,0.01,0.01,0.01,])
        #Now all the plotting of our figures

        fig,(ax1,ax2) = plt.subplots(2,1)
        #First subplot
        ax1.plot(bits, func(bits,*popt), 'r-',label ='Fitted Number Voltage Relationship')
        ax1.plot(bits,volts,'o',label = 'Data Points from Multimeter')
        ax1.errorbar(bits,volts,yerr = error,fmt = '.',label = 'Error Bars')
        ax1.set_title('Voltage Output for 8-bit Values', fontsize = 14)
        ax1.set_xlabel('8-Bit values')
        ax1.set_ylabel('Voltage [V]' )
        ax1.legend()
        ax1.set_facecolor((.97,.97,.97))
        perr = np.sqrt(np.diag(pcov))

        #Second subplot residuals
        ax2.plot(bits,residualNumbers(volts,func(bits, *popt)), 'r.', label = "Voltage Output Residuals")
        ax2.axhline(0,color='k')
        ax2.set_title('Voltage Residuals',fontsize = 14)
        ax2.axis(([-5,257,-0.2,0.2]))
        ax2.set_xlabel('8 Bit Values')
        ax2.set_ylabel('Residuals')
        ax2.legend()
        ax2.set_facecolor((.97,.97,.97))

        #Figure
        fig.align_ylabels()
        fig.tight_layout()
        fig.savefig('LinearFit')






    #Ploting our figures
    figure1()
    figure2()
    linearCalibration()
    #Our figure of A2
    '''
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
    plt.ylabel('Voltage')
   
    plt.savefig('A2')

    '''
    #Subplots, properly
    #Creates a figure and only one subplot
    fig,ax = plt.subplots()
    ax.plot(A2[0],A2Residuals)
    fig.savefig('subplotExample')
    
    #Creates two subplots
    fig, (ax1,ax2) = plt.subplots(2,1)
    ax1.plot(A1[0],yAxis,'o')
    ax2.plot(A1[0],A1Residuals)
    ax1.errorbar(A2[0],A2Residuals,yerr =A2ResidualError*100, fmt = '.')
    ax1.set_ylabel('Voltage')
    ax2.set_ylabel('Voltage')
    fig.align_ylabels()
    fig.savefig('subplotExample2')


    return
main()
