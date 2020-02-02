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
    #Plot and residuals, therefore two subplots
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
    #Ploting our figures
    figure1()
    figure2()
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