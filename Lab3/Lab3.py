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
def section5_3(confidenceInterval):
    ''' Our function running everything we need for section 5.3
    Parameters
    ----------
    Data: Data loaded from a file
    binNum: Number of bins to be used 
    condifenceInterval: The % confidence we are using for the chi square test.
    '''

    #Testing distributions with diffre
    def MC_distiribution_tester(numRandoms, randomMeans):
        ''' Tests poisson and gaussian distributions against randomly generated poisson 
            data
        Parameters
        ----------
        numRandoms: Number of random data points generated
        randomMeans: The random mean the points are generated around
        Returns
        -------
        threshholds : Gaussian and Poisson chi square threshold values
        chiSquare: Gaussian and Poisson chi square values
        '''
        #Generating test data
        testData = np.random.poisson(randomMeans,numRandoms)
        testData = testData[testData > 0]


        #Make the bins only reach 3 standard deviations
        std = np.std(testData)
        mean = np.mean(testData)
        maxBinValue = int(mean + 3*std)
        minBinValue = int(mean - 3*std)
        if mean - 3*std < 0:
            minBinValue = 1

        #Now we cut off the data above our bin
        testData = testData[testData < maxBinValue]

        #Bin Data
        #Number of bins will be equal to max(num)-min(num)
        binNum = maxBinValue - minBinValue

        binnedData,binEdges = np.histogram(testData,binNum,range = (minBinValue,maxBinValue))
        
        
        #Getting bin centers
        binCenter = histogramBinCenters(binEdges)
        for i in range(len(binCenter)):
            binCenter[i] = int(binCenter[i])
        print(binCenter)
        
        #Generating the expectedValues for Poisson and Gaussian distributuions at given values
        #Generating Poisson
        poissonExpected = []
        for i in binCenter:
            poissonExpected.append(poisson(randomMeans,i)*numRandoms)
        #Generate Gaussian
        #Standard Deviation of Datapoints
        std = np.std(testData,ddof = 1)
        gaussianExpected= []
        for i in binCenter:
            gaussianExpected.append(gaussian(i,randomMeans,std)*numRandoms)

        #Time to test the chi square values
        gaussianChiSquareValue = reduced_chi_squared_value_calculator(gaussianExpected,binnedData, binNum-3)
        poissonChiSquareValue = reduced_chi_squared_value_calculator(poissonExpected,binnedData,binNum-2)

        #Get the threshold chi square value
        #With the reduced value, i.e the divisions
        gaussianThreshhold = (chi2.isf(confidenceInterval,binNum -3))/(binNum -3)
        poissonThreshhold = chi2.isf(confidenceInterval,binNum -2)/(binNum -2)



        #Are the distributions consistent
        print("N = " + str(numRandoms))
        print('Means = ' +str(randomMeans))
        #Gaussian
        if gaussianChiSquareValue > gaussianThreshhold:
            print('Gaussian Distribution is not consistent with the data')
        elif gaussianChiSquareValue <= gaussianThreshhold:
             print('Gaussian Distribution is consistent with the data')
             
        #Poisson 
       
        if poissonChiSquareValue > poissonThreshhold:
            print('Poisson Distribution is not consistent with the data')
            
        elif poissonChiSquareValue <= poissonThreshhold:
            print('Poisson Distribution is consistent with the data')

        
        #Data to return
        thresholds = [gaussianThreshhold,poissonThreshhold]
        chiValues = [gaussianChiSquareValue,poissonChiSquareValue]



        #Plot it to check
        '''
        
        plt.show()
        '''
        return thresholds,chiValues

    #Outer function

    #Show how Chi square values evolve with N
    
    numPoints = 70
    Ns = np.linspace(10,1000,numPoints)

    
    for i in range(numPoints):
        Ns[i] = int(Ns[i])

    print(Ns)

    
    Means = [2,3,7]

    #Each list will hold an array of chi square values at a certain mean with different Ns
    gaussianChiSquares = []
    poissonChiSquares = []
    #Thresholds are the maximum chi squared values we can get
    thresholds =[]
    numberToMean = 120
    #Accept distribution string

    for i in Means:
        gaussianAdded = []
        poissonAdded = []
        for j in Ns:
            toMean = []
            for x in range(numberToMean):
                thresholds2, toAdd = MC_distiribution_tester(int(j),i)
                toMean.append(toAdd)

            toMean = np.asarray(toMean).T

            toAdd = [0,0]
            toAdd[0] = np.mean(toMean[0])
            toAdd[1] = np.mean(toMean[1])
            gaussianAdded.append(toAdd[0])
            poissonAdded.append(toAdd[1])
            thresholds.append(thresholds2)
        gaussianChiSquares.append(gaussianAdded)
        poissonChiSquares.append(poissonAdded)
        plt.plot(Ns,gaussianAdded,'.',label = 'Gaussian, Mean = ' + str(i))

        plt.plot(Ns,poissonAdded, '.', label = 'Poisson, Mean = ' +str(i))
        

        
    thresholds = np.asarray(thresholds).T

    #Plot of evolution of chi square values
    plt.title('Evolution of Chi Squared Values')
    plt.xlabel('Number of Samples (N)')
    plt.ylabel("Chi Square Value , $\chi^2$")
    
  
    plt.legend()
    plt.show()
   


    #Means
  

    



    return

#Importing the data
def DataTester(numValues,randomMeans,data,confidenceInterval):
        ''' Tests poisson and gaussian distributions against randomly generated poisson 
            data
        Parameters
        ----------
        numVAlues: Number of random data points generated
        randomMeans: The random mean the points are generated around
        Returns
        -------
        threshholds : Gaussian and Poisson chi square threshold values
        chiSquare: Gaussian and Poisson chi square values
        '''
        #Generating test data
        numRandoms = numValues
        testData = data
        testData = testData[testData > 0]


        #Make the bins only reach 3 standard deviations
        std = np.std(testData)
        mean = np.mean(testData)
        maxBinValue = int(mean + 3*std)
        minBinValue = int(mean - 3*std)
        if mean - 3*std < 0:
            minBinValue = 1

        #Now we cut off the data above our bin
        testData = testData[testData < maxBinValue]

        #Bin Data
        #Number of bins will be equal to max(num)-min(num)
        binNum = maxBinValue - minBinValue

        binnedData,binEdges = np.histogram(testData,binNum,range = (minBinValue,maxBinValue))
        
        
        #Getting bin centers
        binCenter = histogramBinCenters(binEdges)
        for i in range(len(binCenter)):
            binCenter[i] = int(binCenter[i])
        print(binCenter)
        
        #Generating the expectedValues for Poisson and Gaussian distributuions at given values
        #Generating Poisson
        poissonExpected = []
        for i in binCenter:
            poissonExpected.append(poisson(randomMeans,i)*numRandoms)
        #Generate Gaussian
        #Standard Deviation of Datapoints
        std = np.std(testData,ddof = 1)
        gaussianExpected= []
        for i in binCenter:
            gaussianExpected.append(gaussian(i,randomMeans,std)*numRandoms)

        #Time to test the chi square values
        gaussianChiSquareValue = reduced_chi_squared_value_calculator(gaussianExpected,binnedData, binNum-3)
        poissonChiSquareValue = reduced_chi_squared_value_calculator(poissonExpected,binnedData,binNum-2)

        #Get the threshold chi square value
        #With the reduced value, i.e the divisions
        gaussianThreshhold = (chi2.isf(confidenceInterval,binNum -3))/(binNum -3)
        poissonThreshhold = chi2.isf(confidenceInterval,binNum -2)/(binNum -2)



        #Are the distributions consistent
        print("N = " + str(numRandoms))
        print('Mean = ' + str(randomMeans))
        #Gaussian
        if gaussianChiSquareValue > gaussianThreshhold:
            print('Gaussian Distribution is not consistent with the data')
        elif gaussianChiSquareValue <= gaussianThreshhold:
             print('Gaussian Distribution is consistent with the data')
             
        #Poisson 
       
        if poissonChiSquareValue > poissonThreshhold:
            print('Poisson Distribution is not consistent with the data')
            
        elif poissonChiSquareValue <= poissonThreshhold:
            print('Poisson Distribution is consistent with the data')

        
        #Data to return
        thresholds = [gaussianThreshhold,poissonThreshhold]
        chiValues = [gaussianChiSquareValue,poissonChiSquareValue]



        #Plot it to check
        '''
        
        plt.show()
        '''
        return thresholds,chiValues



def main():
    '''Our main method'''
    #Importing the data
    data = np.loadtxt(r"data\count1.csv",delimiter =',')

    #Section 5.3
    section5_3(0.05)


    #Testing the means of 7
    data = np.loadtxt(r"C:\Users\Andrew\Desktop\Classes\PHYS 339\Lab3\count3.csv" , delimiter =',')
    data = data[0]
    print(data)
    mean = np.mean(data)
    DataTester(len(data),mean,data,0.05)


    return


main()
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
'''
plt.hist(data, normed = True , bins =15)
plt.plot(xAxis,yAxis)
'''
#Gaussian 

var = 0
for i in range(len(binData)):
    var = var + (binData[i])*(binCenters[i] - mean)**2

var = var / len(data)
var = np.sqrt(var)
print(var)
print(np.std(binCenters,))
yAxis = gaussian(xAxis,mean,var )
'''
plt.plot(xAxis,yAxis)
plt.show()
'''
print(len(xAxis))

