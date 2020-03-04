#Importing our modules
import numpy as np 
import matplotlib.pyplot as plt 
from scipy import optimize as opt


#==========================================================
#FUNCTIONS
#==========================================================
def malusLawI(theta,V0,phi):

    theta = theta*0.0174533
    phi = phi*0.0174533
    return V0*(np.cos(theta+phi))**2


def malusFitter(voltageData,variance,initialPhiGuess,iterationNum):
    '''Fits Malus' law to a set of data 
    Params:
    -------
    voltageData: our raw voltage data
    variance: the variance on a given data point
    initialPhiGuess: our original guess for phi
    iterationNum: the number of iterations we we will be undergoing to sole

    Returns:
    --------
    phi: a fitted value for phi
    V0 : a fitted value for V0
    '''



    

    #More functions to fit, from appendix B

    def X2P(voltageData,V0,phi,variance):
        '''Returns the value of X2/P
        Params:
        -------
        voltageData The voltage data taken in the experiment
        V0 : Our initial voltage, replaced for I0 in malus' law
        phi: the phase shift between our two polarisers
        variance: the variance in our data, think sigma**2

        Returns:
        --------
        X2P 
        
        '''
        #Sum we want to return
        returnedSum = 0
        #Steps go off the assumption that 1 degree = 1 step
        #We multiply by 0.0174533 to get radians
        steps = 1
        for i,j in zip(voltageData,variance):
            returnedSum += ((i - V0*np.cos(steps*0.0174533 + phi*0.0174533)**2)*np.cos(steps*0.0174533 + phi*0.0174533)*np.sin(steps*0.0174533 + phi*0.0174533))/j
            steps += 1
        return returnedSum*4*V0

    def X2P2(voltageData,V0,phi,varaince):
        """ Returns the value of X2/P2
        Params:
        -------
        voltageData The voltage data taken in the experiment
        V0 : Our initial voltage, replaced for I0 in malus' law
        phi: the phase shift between our two polarisers
        variance: the variance in our data, think sigma**2, it's an array

        Returns:
        --------
        X2P2
        """
        #Defining our initial sums for the calculations
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        #Steps go off the assumption that 1 degree = 1 step
        #We multiply by 0.0174533 to get radians
        steps = 1
        for i,j in zip(voltageData,variance):
            #Our angle to put inside the cos and sin functions, in rads
            angle = steps*0.0174533 + phi*0.0174533

            #Updating our sums
            sum1 += (np.cos(angle)**2)/j
            sum2 += (np.cos(angle)**4)/j
            sum3 += i/j
            sum4 += i*(np.cos(angle)**2)/j

            #Updating steps
            steps += 1

        return (12*V0**2)*sum1 -(16*V0**2)*sum2 - 4*V0*sum3 + (8*V0)*sum4




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
        for i,j in zip(voltageData,variance):
            numerator += (i*(np.cos(steps*0.0174533+phi*0.0174533))**2)/j
            denominator += (np.cos(i*0.0174533+phi*0.0174533))**4/j
            steps += 1

        return numerator/denominator

    def newPhi(oldPhi,voltageData,V0,varaince):
        '''Returns an estimation of phi based on the old value of phi 
        Params:
        -------
        oldPhi: the previous guess for phi
        voltageData: the voltage data we are fitting to
        V0: guess for V0
        
        variance: the variance in our data, think sigma**2

        Returns:
        --------
        A new value of phi
         '''

        returnedValue = oldPhi - X2P(voltageData,V0,oldPhi,varaince)/X2P2(voltageData,V0,oldPhi,varaince)
        return returnedValue


    #------------------------------------------------
    returnedPhis = []
    returnedV0s = []
    #initalGuess for V0
    phi = initialPhiGuess
    V0Val = V0(voltageData,phi,variance) 

   
    #Now move to finding the values
    for i in range(iterationNum):
        returnedV0s.append(V0Val)
        returnedPhis.append(phi)
        newphi = newPhi(phi,voltageData,V0Val,variance)
        phi = newphi
        V0Val = V0(voltageData,phi,variance)

        if(i%100 == 0):
            print(i)
            print(V0Val)


    


    return returnedPhis,returnedV0s


#==========================================================
#Section 2.1 - Malus' Law
#==========================================================

#Importing that CSV 
Lab4steps = np.loadtxt(r"Lab4steps.csv",delimiter = ',')
Lab4Vals = np.loadtxt(r"Lab4Vals.csv", delimiter = ',')
data21 = np.loadtxt(r"AndysVoltageVals.csv", delimiter = ',')

#Data Analysis
#-------------

#Finding the max value to normalize the data
maxVal = 0
for i in data21:
    
    for j in i:
        if j > maxVal:
            maxVal = j
        
#Normalizing the data
for i in range(len(data21)):

    for j in range(len(data21[i])):
        data21[i][j] = data21[i][j]/maxVal
 
#Getting the x range of the device
xs = np.linspace(0,len(data21[0]) -1, len(data21[0]))

#get the means and std of the data
means = []
std = []
for i in data21.T :
    means.append(np.mean(i))
    std.append(np.std(i))

#Fitting our function
values,cov  = opt.curve_fit(malusLawI,xs,means)
vnaught = values[0]
phi = values[1]


#======================================================================
#Section 2,3
#======================================================================
#Load all of the data
brewsterData = np.loadtxt(r"maybe_brewster_BBD.csv", delimiter = ',')
xAxis = np.linspace(0,len(brewsterData)-1, len(brewsterData) )

#We are looking for a maximum in the transmitted light without the parts where the beam
#goes through the glass




#======================================================================
#FIGURES
#======================================================================

#FIG1 -Malus' Law with Residuals
def Fig1():
	fig, (ax1,ax2) = plt.subplots(2,1, constrained_layout = True)
	fig.suptitle("        Fitting of Malus' Law to Observed Data")
	ax1.plot(xs,malusLawI(xs,vnaught,phi), label = "Fitted Curve of Malus' Law")
	ax1.errorbar(xs,means,yerr = std, fmt =  '.', label = "Voltage Response of Photodiode")
	ax1.set_ylabel('Photodiode Response')
	ax1.set_xlabel('Angle of Polariser')
	ax1.set_facecolor((0.97,0.97,0.97))
	ax1.legend(loc = 'lower right')
	ax1.set_title('Voltage Response vs. Angle of Polariser')

	#Residuals
	ax2.errorbar(xs,means - malusLawI(xs,vnaught,phi)  , yerr = std, fmt = '.', label = 'Residuals of Fit')
	ax2.set_ylabel('Residual')
	ax2.set_xlabel('Angle of Polariser')
	ax2.set_facecolor((0.97,0.97,0.97))
	ax2.legend(loc = 'lower right')
	ax2.set_title("Residuals of Fitting Malus' Law to Observed Data")
	ax2.axhline(0,c ='black')



	fig.align_ylabels()
	fig.savefig("Malus'_Law")
	plt.close()
	return

#FIG2 - Brewster's Angle

def Fig2():
	plt.plot(xAxis,brewsterData)
	plt.show()
	return


#-----------------
#Running the Figures
Fig2()




