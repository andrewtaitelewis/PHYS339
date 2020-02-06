#Importing all the relevant modules 
import math as m
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import chi2

#Chi Squared Functions -- IMPORTED LAB 1 --
def chi_squared_value_calculator(expected_behavior, observed_behavior):
    """ Calculates the chi square value for a given set of expected and observed behavior
    Parameters
    ----------
    expected_behavior: The expected behavior to be testing against. 1D Array
    observed_behavior: The behavior observed in the experiment. 1D Array

    Returns
    -------
    int chiSquareValue: The chi square value that was calculated
    """
    chiSquareValue = 0
    for i in range(len(expected_behavior)):
        chiSquareValue += ((observed_behavior[i]-expected_behavior[i])**2)/expected_behavior[i]


    print('Chi Square Value is')
    print(chiSquareValue)
    return chiSquareValue


def pValueCalculator(chiSquareValue, dof):
    """ Accepts or rejects the hypothesis based on a chi squared value and a pValue
    Parameters
    ---------
    chiSquareValue: The chi squared value calculated for this given experimet
    pValue        : The p value that you will be calculated against, example 0.05

    Returns
    -------
    boolean True, False, True = accepts null hypothesis, False = rejects null hypothesis
    """

    value = 1 - chi2.cdf(chiSquareValue,dof)
    print(value)
    return value


def chiSquareTest(expected_behavior, observed_behavior, confidenceInterval):
    """ Accepts or rejects the null hypothesis
    Parameters
    ----------
    expected_behavior: The expected behavior to be testing against. 1D Array
    observed_behavior: The behavior observed in the experiment. 1D Array
    confidenceInterval: The p value we are comparing against

    Returns
    -------
    Boolean True, reject the null hypothesis. False, accept the null hypothesis
     """

    chiSquareValue = chi_squared_value_calculator(expected_behavior,observed_behavior)
    dof = len(expected_behavior) -1
    pValue = pValueCalculator(chiSquareValue,dof)

    if(pValue < confidenceInterval):
        print('Reject null hypothesis')
        return True
    if(pValue >= confidenceInterval):
        print('Accept null hypothesis')
        return False

#Chi Squared Functions --EDITED LAB 2 --

def reduced_chi_squared_value_calculator(expected_behavior,observed_behavior,dof):
    ''' Calculates the reduced chi square value, using the chi square value and degrees of freedom
    Parameters
    ----------
    expected_behavior: The expected behavior of our given distribution
    observed_behavior: The behavior that was observed in the lab
    dof: The degrees of freedom for the given distribution
    '''

    chiSquareValue = chi_squared_value_calculator(expected_behavior,observed_behavior)
    return chiSquareValue/dof
def poisson(u,n):
    return np.exp(-u)*(u**(n))/m.factorial(n)

def gaussian(x, mu, sig):
    """Our Gaussian distribution"""
    return 1/(sig*np.sqrt(2*np.pi))*np.exp((-1/2)*np.power((x - mu)/sig, 2))

def histogramBinCenters(xBinValues):
    """ Returns an array of bin ceneters given bin edges
    Parameters
    ----------
    xBinValues: The edges of all the histogram bins

    Returns
    -------
    Array of the center of the histogram bins"""
    returnedArray = []
    length = (len(xBinValues))
    length = length -1


    for i in range(length) :
        returnedArray.append((xBinValues[i] + xBinValues[i+1])/2)
    return returnedArray

#Section 5.3
#We will need to vary the chi square distributions
def section5_3(data, binNum,confidenceInterval):
    ''' Our function running everything we need for section 5.3
    Parameters
    ----------
    Data: Data loaded from a file
    binNum: Number of bins to be used 
    condifenceInterval: The % confidence we are using for the chi square test.
    '''
    #Helper Functions
    def gaussian_tester(expected_behavior, observed_behavior,binNum):
        '''Tests the data against the gaussian distribution 
        Parameters
        ----------
        expected_behavior: The expected behavior of the distribution
        observed_behavior: The data that was observed in the experiment
        binNum: The number of bins used to bin the data
        Returns
        -------
        Reduced Chi square value 
        '''
        #Degrees of freedom for the distribution
        dof = binNum - 3

        reducedChiSquare = reduced_chi_squared_value_calculator(expected_behavior, observed_behavior, dof)   
        return reducedChiSquare

    def poisson_tester(expected_behavior, observed_behavior,binNum):
        '''Tests the data against the gaussian distribution 
        Parameters
        ----------
        expected_behavior: The expected behavior of the distribution
        observed_behavior: The data that was observed in the experiment
        binNum: The number of bins used to bin the data
        Returns
        -------
        Reduced Chi square value 
        '''
        dof = binNum - 2

        reducedChiSquare = reduced_chi_squared_value_calculator(expected_behavior, observed_behavior, dof)

        return reducedChiSquare

    def MC_distiribution_tester(numRandoms, randomMeans):
        ''' Tests poisson and gaussian distributions against randomly generated poisson 
            data
        Parameters
        ----------
        numRandoms: Number of random data points generated
        randomMeans: The random mean the points are generated around
        '''
        #Generating test data
        testData = np.random.poisson(randomMeans,numRandoms):

        #Bin Data
        binnedData,binEdges = np.histogram(data,binNum)

        #Getting bin centers
        binCenter = histogramBinCenters(binEdges)
        for i in binCenter:
            binCenter[i] = int(i)
        #Generating the expectedValues for Poisson and Gaussian distributuions at given values
        #Generating Poisson
        poissonExpected = []
        for i in binCenter:
            poissonExpected.append(poisson(randomMeans,i))
        #Generate Gaussian
        #Standard Deviation of Datapoints
        std = np.std(testData,ddof = 1)
        gaussianExpected= []
        for i in binCenter:
            gaussianExpected.append(gaussian(i,randomMeans,std))

        #Time to test the chi square values
        gaussianChiSquareValue = gaussian_tester(gaussianExpected,binnedData, binNum)
        poissonChiSquareValue = poisson_tester(poissonExpected,binnedData,binNum)

        #Get the threshold chi square value



        

        return

    #Outer function

    return


#Importing the data
def main():
    '''Our main method'''
    #Importing the data
    data = np.loadtxt(r"data\count1.csv",delimiter =',')

    #Section 5.3
    section5_3(data,4)

    return
data = np.loadtxt(r"data\count1.csv",delimiter =',')
binData,binEdges, dataType = plt.hist(data,bins = 15)
plt.close()
#Return the bin centers
binCenters = histogramBinCenters(binEdges)

#Turn the binCenters into ints
for i in range(len(binCenters)):
    binCenters[i] = int(binCenters[i])

#Get the center of all the bins
mean = np.mean(data)
#sum(binCenters*binData)/len(data)

#Our Poisson Data
xAxis = np.linspace(binCenters[0],binCenters[-1], -binCenters[0] +binCenters[-1] +1)
yAxis = []
for i in xAxis:
    yAxis.append(poisson(mean,i))

plt.hist(data, normed = True , bins =15)
plt.plot(xAxis,yAxis)

#Gaussian 

var = 0
for i in range(len(binData)):
    var = var + (binData[i])*(binCenters[i] - mean)**2

var = var / len(data)
var = np.sqrt(var)
print(var)
print(np.std(binCenters,))
yAxis = gaussian(xAxis,mean,var )
plt.plot(xAxis,yAxis)
plt.show()
print(len(xAxis))