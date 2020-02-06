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
    '''

    chiSquareValue = chi_squared_value_calculator(expected_behavior,observed_behavior)

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



#Histogram
#Importing the data
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