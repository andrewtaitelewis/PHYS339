#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 

#Importing that CSV 
Lab4steps = np.loadtxt(r"C:\Users\Andrew\Desktop\Classes\PHYS 339\Lab4\Lab4steps.csv",delimiter = ',')
Lab4Vals = np.loadtxt(r"C:\Users\Andrew\Desktop\Classes\PHYS 339\Lab4\Lab4Vals.csv", delimiter = ',')


#Our Functions 
def malusLawI(I0,theta):

    return I0*(np.cos(theta))**2


def malusFitter():
    '''Okay mhmmm this is gonna be a doozie '''

    #More functions to fit, from appendix B

    def X2P(voltageData,V0,phi):
        '''Our Chi Square partial with respect to V0
        Params:
        -------
        voltageData The voltage data taken in the experiment
        V0 : Our initial voltage, replaced for I0 in malus' law
        phi: the phase shift between our two polarisers
        
        Returns:
        --------
        X2P 
        
        '''
        #Sum we want to return
        returnedSum = 0
        #Steps go off the assumption that 1 degree = 1 step
        #We multiply by 0.0174533 to get radians
        steps = 1
        for i in voltageData:
            returnedSum += (i - V0*np.cos(steps*0.0174533 + phi*0.0174533)**2)*np.cos(steps*0.0174533 + phi*0.0174533)*np.sin()
        



        return

    def V0(voltageData,phi,variance):
        '''Calculates V0
        Params
        ------
        voltageData: the voltage data we are fitting to
        phi: the initial angle difference between our polarisers
        variance: the variance in our data, think sigma**2

        Returns
        -------
        Estimate of V0

        '''
        #Numerator and Denominator of our expression
        numerator = 0
        denominator = 0
        #Steps goes off the assumption that 1 step = 1 degree
        steps = 1

        #The reason for the multiplication by 0.0174533 is because np.cos uses rads
        for i in voltageData:
            numerator += (i* np.cos(steps*0.0174533+phi*0.0174533)**2)/variance
            denominator += (np.cos(i*0.0174533+phi*0.0174533)**4)/variance
            steps += 1

        return numerator/denominator

    def newPhi(oldPhi):


    return

#----------------------------------

#Section 2.1 - Malus' Law

#X and Y axis
xAxis = Lab4steps
yAxis = Lab4Vals

#Normalize the yAxis
yNormalized = yAxis/(max(yAxis))

#Our Malus's Law     1 Step = 1 Degree
malusLaw = malusLaw(0.1,xAxis)
#Plot the figure
plt.plot(xAxis,yNormalized)
plt.plot(xAxis,malusLaw)
plt.show()

