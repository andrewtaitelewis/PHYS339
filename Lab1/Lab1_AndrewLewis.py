#Import all of our modules
import numpy as np
import random as r
import matplotlib.pyplot as plt
from scipy.stats import chi2


#Uniform Probability Distribution
def uniform_distribution(number_of_runs):
    """Returns a uniform distribution:
    **number_of_runs**: number of numbers generated in the probability distribution
    **Returns**: An array of values from the uniform probability distribution
    """

    returnedArray = []
    for i in range(number_of_runs):
        returnedArray.append(r.random())
    return returnedArray
#Linear Probability Distribution
def linear_distribution(number_of_runs):
    """ Returns a Linear Distribution between 0 and 1
        parameters
        ----------
        **number_of_runs**: number of numbers generated in the probability distribution
        Returns
        -------
        float array:
        an array with numbers from the probability distribution"""
    returnedArray = []
    for i in range(number_of_runs):
        returnedArray.append(np.sqrt(r.random()))
    return returnedArray
#Gaussian Probability Distribution
def gaussian_distribution(number_of_runs,variance):
    """ Gaussian Distribution between 0 and 1

        Parameters
        ---------
        number_of_runs: number of numbers generated in the probability distribution
        variance: Standard deviation of the Gaussian Distribtution (sigma)

        Returns
        -------
        float array:
        An array of values based on a Gaussian Distribution
          """

    #We will use the linear distribution and the exponential distribution 

    returnedArray = []
    for i in range(number_of_runs):
        returnedArray.append(np.sqrt(-2*variance**2*np.log(1-r.random()))*np.cos(uniform_distribution(1)[0]*2*np.pi))
    return returnedArray
#Gaussian without random
def gaussian(x, mu, sig):
    """Our Gaussian distribution"""
    return 1/(sig*np.sqrt(2*np.pi))*np.exp((-1/2)*np.power((x - mu)/sig, 2))
#Our error bars function
def hist_error_Bars(num_bins,histValues):
    """ Returns the error bar, (standard deviation),  for a certain number of bins
    Parameters
    ----------
    num_bins  : The number of bins in our historgram
    histValues: 2d array of measurements for the histogram error bars 
    Returns
    -------
    An array of standard deviations starting at bin 0, and going to bin max
     """   
    StdArray = [] #Array of standard deviations for each bin
    
    for i in range(num_bins):
        StdArray.append(np.std(histValues[i], ddof = 1))
    return StdArray
#Chi squared test for significance
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

#Our main method
def main(numSamples,numBins, numTrials):
    """
    The main method for Lab 1
    Parameters:
    ----------
    numSamples: Number of samples in each histogram distribution
    numBins   : Number of bins in each histogram 
    numTrials : Number of histogram distributions calculated
    Returns:
    Nothing.

    """
    #Calculating Distributions
    #--------------------------------------------------------------------------
    #Initalzing histogram distribution
    linearHistDistribution = []
    uniformHistDistribution = []
    gaussianHistDistribution = []
    #Getting the different histogram bins of each distribution
    print('Calculating the distributions...')
    for i in range(numTrials):
        Linear_Values = linear_distribution(numSamples)
        Uniform_Values = uniform_distribution(numSamples)
        Gaussian_Values = gaussian_distribution(numSamples, 1)
        #Getting the histogram bins
        LinearHistValues = np.histogram(Linear_Values,numBins,density = True)[0]
        UniformHistValues = np.histogram(Uniform_Values, numBins,density = True)[0]
        GaussianHistValues = np.histogram(Gaussian_Values,numBins, density = True)[0]
        #Appending the Hist Values to our distribution
        linearHistDistribution.append(LinearHistValues)
        uniformHistDistribution.append(UniformHistValues)
        gaussianHistDistribution.append(GaussianHistValues)

    #Change lists to np arrays so we can transpose 
    print('Lists to arrays')
    linearHistDistribution = np.array(linearHistDistribution)
    uniformHistDistribution = np.array(uniformHistDistribution)
    gaussianHistDistribution = np.array(gaussianHistDistribution)
    
    #Transpose
    print('Transposing the lists')
    linearHistDistribution = np.transpose(linearHistDistribution)
    uniformHistDistribution = np.transpose(uniformHistDistribution)
    gaussianHistDistribution = np.transpose(gaussianHistDistribution)
    
    #Calculating the error bars on each histogram
    print('Calculating the error bars...')
    linearErrorBars = hist_error_Bars(numBins,linearHistDistribution)
    uniformErrorBars = hist_error_Bars(numBins,uniformHistDistribution)
    gaussianErrorBars = hist_error_Bars(numBins,gaussianHistDistribution)

    #Calculating Probability Distributions
    #----------------------------------------------------------------

    # Probability Distribution
    Linear_Values = linear_distribution(numSamples)
    Uniform_Values = uniform_distribution(numSamples)
    Gaussian_Values = gaussian_distribution(numSamples, 1)

    #Our distributions we will be testing against 
    #Gaussian
    xGauss =         np.linspace(-4,4)
    gaussianTest = gaussian(xGauss,0,1)

    #Figures
    #--------------------------------------------------------------------
    #Linear Distribution histogram
    f1 = plt.figure()
    plotHistogram = plt.hist(Linear_Values, numBins,density= True, stacked = True)
    
    #Fit values to a line x = y
    x = np.linspace(0,1)
    plt.plot(x,2*x)  #Normalized
    

    #Errorbars
    xBin = plotHistogram[1]
    yBin = plotHistogram[0]
    xBin = histogramBinCenters(xBin)
    linearErrorBars = np.array(linearErrorBars)
    plt.errorbar(xBin,yBin,yerr = linearErrorBars, fmt = 'ro')

    plt.legend(['Linear Probability Distribution','Histogram Counts of Generated Distribution','Error Bars on Histogram Bins',])
    plt.title('Linear Probability Distribution')
    plt.xlabel('Values')
    plt.ylabel('Counts per Bin, Normalized')

    plt.savefig('LinearHistogram')
    plt.close()

    #Linear Chi Squared Test
    xBin = np.array(xBin)
    linearProbability = (2*xBin)
    
    #Running chi sqaure
    linearProbability = np.array(linearProbability)
    print('\nLinear Chi Squared Test:')
    print('--------------------------')
    chiSquareTest(linearProbability,yBin,0.01)


    #Uniform Distribution Histogram
    
    f2 = plt.figure()
 
    plotHistogram = plt.hist(Uniform_Values,numBins,density=True, stacked= True)
    plt.xlabel('Values')
    plt.ylabel('Counts')

    #ErrorBars
    xBin = plotHistogram[1]
    yBin = plotHistogram[0]
    xBin = histogramBinCenters(xBin)
    plt.errorbar(xBin,yBin,yerr = uniformErrorBars, fmt = 'ro')
    plt.plot(x,np.ones(50))
    plt.legend(['Uniform Probability Distribution','Histogram Counts of Generated Distribution','Error Bars on Histogram Bins',])
    plt.title('Uniform Probability Distribution')
    plt.xlabel('Values')
    plt.ylabel('Counts per Bin, Normalized')
    plt.savefig('UniformHistogram')
    plt.close()

    #Uniform Chi Squared Test
    print('\nUniform Chi Squared Test:')
    print('--------------------------')
    uniformProbability = np.ones(20) 
    
    chiSquareTest(uniformProbability,yBin,0.01)
    #Histogram of Gaussian Values
   

    f3 = plt.figure()
    plotHistogram = plt.hist(Gaussian_Values,numBins,density=True, stacked= True)
    xBin = plotHistogram[1]
    yBin = plotHistogram[0]
    xBin = histogramBinCenters(xBin)
    
    plt.xlabel('Values')
    plt.ylabel('Counts')
    #Plot a Gaussian over it
    plt.plot(xGauss,gaussianTest)
    #Plot the error bars
    
    plt.errorbar(xBin,yBin,yerr = gaussianErrorBars, fmt = 'ro')
    plt.legend(['Probability Distribution','Histogram Counts','Error Bars',])
    plt.title('Gaussian Probability Distribution')
    plt.xlabel('Values')
    plt.ylabel('Counts per Bin, Normalized')
    plt.savefig('GaussianHistogram')
    plt.close()

    #Gaussian Chi Squared Test
    print('\nGaussian Chi Squared Test:')
    print('-------------------------- \n')
    xBin = np.array(xBin)
    GaussianProbability = gaussian(xBin,0,1)
    
    #Running chi sqaure
    chiSquareTest(GaussianProbability,yBin,0.01)

    #a little bet
    maxValue = 0
    for i in range(1000000):
        a = gaussian_distribution(1,1)[0]
        if a > maxValue:
            maxValue = a
    
    print(maxValue)
main(100000,20,10)

"""
Below this point is the code used to simulate the evolution of the means and stds of each dist

Created on Sun Jan 12 14:24:57 2020

@author: gabriel_giampa
"""

import numpy as np
import matplotlib.pyplot as plt
import random as r

def lin(ro):
    return np.sqrt(ro)

sig = 1
var = sig**2
def gauss(ro):
    tval = r.uniform(0,2*np.pi)
    rval = np.sqrt(-2*var*np.log(1 - ro))
    return rval * np.cos(tval)
    

N = 10 #initial number of samples

iterations = 100 #increase sample size 99 times
samples = np.zeros(iterations)
samples[0] = 10
uni_mean = np.zeros(iterations)
lin_mean = np.zeros(iterations)
gaus_mean = np.zeros(iterations)
uni_std = np.zeros(iterations)
lin_std = np.zeros(iterations)
gaus_std = np.zeros(iterations)

for i in range(iterations):
    uni_val = np.zeros(int(samples[i]))
    lin_val = np.zeros(int(samples[i]))
    gaus_val = np.zeros(int(samples[i]))
    
    for j in range(int(samples[i])):
        ran = r.random()
        uni_val[j] = ran
        lin_val[j] = lin(ran)
        gaus_val[j] = gauss(ran)
    
    uni_mean[i] = np.mean(uni_val)
    uni_std[i] = np.std(uni_val)
    lin_mean[i] = np.mean(lin_val)
    lin_std[i] = np.std(lin_val)
    gaus_mean[i] = np.mean(gaus_val)
    gaus_std[i] = np.std(gaus_val)
    
    if i != 99:
        samples[i+1] = samples[i] + 400



plt.figure()
plt.title("Evolution of Uniform Distribution Mean as Sample Size Grows")
plt.xlabel("Sample Size")
plt.ylabel("Mean")
plt.plot(samples,uni_mean, 'o')
plt.show()

plt.figure()
plt.title("Evolution of Linear Distribution Mean as Sample Size Grows")
plt.xlabel("Sample Size")
plt.ylabel("Mean")
plt.plot(samples, lin_mean, 'o')
plt.show()

plt.figure()
plt.title("Evolution of Gaussian Distribution Mean as Sample Size Grows")
plt.xlabel("Sample Size")
plt.ylabel("Mean")
plt.plot(samples,gaus_mean, 'o')
plt.show()

plt.figure()
plt.title("Evolution of Uniform Distribution St. Deviation")
plt.xlabel("Sample Size")
plt.ylabel("STD")
plt.plot(samples,uni_std, 'o')
plt.show()

plt.figure()
plt.title("Evolution of Linear Distribution St. Deviation")
plt.xlabel("Sample Size")
plt.ylabel("STD")
plt.plot(samples,lin_std, 'o')
plt.show()

plt.figure()
plt.title("Evolution of Gaussian Distribution St. Deviation")
plt.xlabel("Sample Size")
plt.ylabel("STD")
plt.plot(samples,gaus_std, 'o')
plt.show()

"""
Below this point is the code used to simulate digitization
Note Ns needs to be manually changed, the program will not loop through an array of sample sizes
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
Created on Sun Jan 19 14:42:45 2020

@author: gabriel_giampa
"""

import numpy as np
import random as r
import matplotlib.pyplot as plt

def gauss(ro, sig):
    var = sig**2
    tval = r.uniform(0,2*np.pi)
    rval = np.sqrt(-2*var*np.log(1 - ro))
    return rval * np.cos(tval) + 0.5

def histMean(sum_range, values, edges):
    binCenter = np.zeros(len(edges) - 1)
    for i in range(len(edges) - 1):
        binCenter[i] = 0.5 * (edges[i] + edges[i+1])
    rollingSum = 0.0
    for i in range(len(values)):
        rollingSum += values[i]*binCenter[i]
    return rollingSum / sum_range

Ns = 10000
sig = 0.001

sigma = np.zeros(100)
sigma[0] = 0.001

values = np.zeros(Ns)
means = np.zeros(10)
uncerts = np.zeros(100)

for k in range(100):
    for i in range(10):
        for j in range(Ns):
            values[j] = gauss(r.random(), sig)
        bin_val, bin_edges = np.histogram(values, 256, (0,1))
        sum_in_range = sum(bin_val) #should be 1000 or lower, if sigma too high
        means[i] = histMean(sum_in_range, bin_val, bin_edges)
    uncerts[k] = np.std(means)
    sig = sig + 0.005
    if(k!= 99):
        sigma[k+1] = sigma[k] + 0.005


plt.figure()
plt.plot(sigma,uncerts)
plt.xlabel("Standard Deviation $\sigma$")
plt.axhline(1/256.0, linestyle = '-', color = 'orange')
plt.ylabel("Uncertainty in mean")
plt.title("Evolution of Uncertainty in List of 10 Means as $\sigma$ Grows, N = " + str(Ns))
plt.show()

