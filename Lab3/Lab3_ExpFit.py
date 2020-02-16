#Importing all the relevant modules 
import math as m
import numpy as np 
import matplotlib.pyplot as plt
from scipy.stats import chi2
from scipy import optimize 
#Our Functions
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

def exponential(data,a,b,c):
    ''' Returns an exponential of A*e^-b '''

    return a*(np.exp(1)**(data*b)) + c

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

#Loading our dwell time data
dwellData = np.loadtxt(r"C:\Users\Andrew\Desktop\Classes\PHYS 339\Lab3\dwellTime.csv", delimiter =',')

#Our confidence interval and number of data points
confidenceInterval = 0.05
numData = 100


#Time between events in microseconds
data = dwellData

#Intializing our empty 
height1 = np.zeros(20)
edges1 = np.zeros(20)
patches1 = np.zeros(20)

#Histogramming a bunch of different bins
h,e,p =  plt.hist(data[0]*10**(-6), bins = 10, range =(0,0.08))
h1,e1,p1 =  plt.hist(data[1]*10**(-6), bins = 10, range =(0,0.08))
h2,e2,p2 =  plt.hist(data[2]*10**(-6), bins = 10, range =(0,0.08))
h3,e3,p3 =  plt.hist(data[3]*10**(-6), bins = 10, range =(0,0.08))
h4,e4,p4 =  plt.hist(data[4]*10**(-6), bins = 10, range =(0,0.08))
h5,e5,p5 =  plt.hist(data[5]*10**(-6), bins = 10, range =(0,0.08))
h6,e6,p6 =  plt.hist(data[6]*10**(-6), bins = 10, range =(0,0.08))
h7,e7,p7 =  plt.hist(data[7]*10**(-6), bins = 10, range =(0,0.08))
h8,e8,p8 =  plt.hist(data[8]*10**(-6), bins = 10, range =(0,0.08))
h9,e9,p9 =  plt.hist(data[9]*10**(-6), bins = 10, range =(0,0.08))
h10,e10,p10 =  plt.hist(data[10]*10**(-6), bins = 10, range =(0,0.08))
h11,e11,p11 =  plt.hist(data[11]*10**(-6), bins = 10, range =(0,0.08))
h12,e12,p12 =  plt.hist(data[12]*10**(-6), bins = 10, range =(0,0.08))
h13,e13,p13 =  plt.hist(data[13]*10**(-6), bins = 10, range =(0,0.08))
h14,e14,p14 =  plt.hist(data[14]*10**(-6), bins = 10, range =(0,0.08))
plt.close()     #Close 

#The bin centers
binCenters = histogramBinCenters(e1)
binCenters = np.asarray(binCenters)
binCenters = np.linspace(1,10,10)


centers = np.zeros(10)
for i in range(10):
    centers[i] = 0.5* (e1[i] + e1[i+1])

centers = binCenters
binMeans = np.zeros(10)
binStds = np.zeros(10)

for i in range(10):
    binMeans[i] = (h[i]+ h1[i] + h2[i] + h3[i] + h4[i] +h5[i]+ h6[i] + h7[i] + h8[i] + h9[i] + h10[i] + h11[i] + h12[i] + h13[i] + h14[i]) / 15
    binStds[i] = np.std([h[i], h1[i], h2[i], h3[i], h4[i],h5[i], h6[i] , h7[i] , h8[i] , h9[i] , h10[i] , h11[i] , h12[i] , h13[i] , h14[i]])
   
#Curve fitting our exponential curve
ppot, pcov = optimize.curve_fit(exponential,binCenters,binMeans,[10,-1,6])
#Our fitted parameters
A,b,c = ppot
A = float(A)
b = float(b)
c = float(c)

#We want to compare poisson and gaussian
binMeans = np.asarray(binMeans)     #Convert to a numpy array
binCenters = np.asarray(binCenters)


variance = 0
#Different mean
weightedMean = np.mean((data[0]*10**-6)*10+1)

for i,j in zip(binCenters,binMeans):
    
    variance += ((i - weightedMean)**2)*j
    
variance = variance/(numData -1)
variance = np.sqrt(variance)


poissonValues = []
gaussianValues = []

for i in binCenters:
    gaussianValues.append(gaussian(i,weightedMean,variance)*100)
    poissonValues.append(poisson(weightedMean,i)*100)

#Plotting our fitted histogram figure
plt.figure()
plt.title("Mean Histogram of 15 Runs")
plt.xlabel(r'Dwell time between each event')
plt.ylabel("Counts")
plt.bar(centers,binMeans,1, yerr = binStds)
plt.plot(centers, exponential(centers,A,b,c),'r',label = 'Exponential')
plt.plot(binCenters,gaussianValues, '.', color = 'orange',label='Gaussian')
plt.plot(binCenters,poissonValues ,'.', color = 'green',label = 'Poisson')
plt.legend()
plt.axvline(weightedMean, color = 'black')
plt.show()

#Test some chi squared values
#Calculate
chiSquareExp = reduced_chi_squared_value_calculator(exponential(centers,A,b,c),binMeans,7)
chiSquareGauss = reduced_chi_squared_value_calculator(gaussianValues,binMeans,7)
chiSquarePos = reduced_chi_squared_value_calculator(poissonValues,binMeans,8)

#Different Thresholds
gaussianThreshhold = (chi2.isf(confidenceInterval,10 -3))/(10 -3)
poissonThreshhold = chi2.isf(confidenceInterval,10 -2)/(10-2)
expThreshold = chi2.isf(confidenceInterval,7)

#Testing exponential
if(chiSquareExp < expThreshold):
    print('Exponential fit is consitient')
elif(chiSquareExp >= expThreshold):
    print('Exponential is not consitent')

#Testing gaussian
if(chiSquareGauss < gaussianThreshhold):
    print('Gaussian fit is consistent ')
elif(chiSquareGauss >= gaussianThreshhold):
    print('Gaussian fit is not consistent')

#Testing poisson
if(chiSquarePos < poissonThreshhold):
    print('Poisson fit is consistent ')
elif(chiSquarePos >= poissonThreshhold):
    print('Poisson fit is not consistent')


def dataChiSquareTester(data, numData):

    #Initalizing our arrays
    height1 = np.zeros(20)
    edges1 = np.zeros(20)
    patches1 = np.zeros(20)

    #Our different histogram bins
    h,e,p =  plt.hist(data[0]*10**(-6), bins = 10, range =(0,0.08))
    h1,e1,p1 =  plt.hist(data[1]*10**(-6), bins = 10, range =(0,0.08))
    h2,e2,p2 =  plt.hist(data[2]*10**(-6), bins = 10, range =(0,0.08))
    h3,e3,p3 =  plt.hist(data[3]*10**(-6), bins = 10, range =(0,0.08))
    h4,e4,p4 =  plt.hist(data[4]*10**(-6), bins = 10, range =(0,0.08))
    h5,e5,p5 =  plt.hist(data[5]*10**(-6), bins = 10, range =(0,0.08))
    h6,e6,p6 =  plt.hist(data[6]*10**(-6), bins = 10, range =(0,0.08))
    h7,e7,p7 =  plt.hist(data[7]*10**(-6), bins = 10, range =(0,0.08))
    h8,e8,p8 =  plt.hist(data[8]*10**(-6), bins = 10, range =(0,0.08))
    h9,e9,p9 =  plt.hist(data[9]*10**(-6), bins = 10, range =(0,0.08))
    h10,e10,p10 =  plt.hist(data[10]*10**(-6), bins = 10, range =(0,0.08))
    h11,e11,p11 =  plt.hist(data[11]*10**(-6), bins = 10, range =(0,0.08))
    h12,e12,p12 =  plt.hist(data[12]*10**(-6), bins = 10, range =(0,0.08))
    h13,e13,p13 =  plt.hist(data[13]*10**(-6), bins = 10, range =(0,0.08))
    h14,e14,p14 =  plt.hist(data[14]*10**(-6), bins = 10, range =(0,0.08))
    plt.close()

    #Bin Centers
    binCenters = np.linspace(1,10,10)
    
    #The mean and Std of bins
    binMeans = np.zeros(10)
    binStds = np.zeros(10)
    for i in range(10):
        binMeans[i] = (h[i]+ h1[i] + h2[i] + h3[i] + h4[i] +h5[i]+ h6[i] + h7[i] + h8[i] + h9[i] + h10[i] + h11[i] + h12[i] + h13[i] + h14[i]) / 15
        binStds[i] = np.std([h[i], h1[i], h2[i], h3[i], h4[i],h5[i], h6[i] , h7[i] , h8[i] , h9[i] , h10[i] , h11[i] , h12[i] , h13[i] , h14[i]])
       
    #Curve fitting the exponential
    ppot, pcov = optimize.curve_fit(exponential,binCenters,binMeans,[10,-1,6])

    A,b,c = ppot
    A = float(A)
    b = float(b)
    c = float(c)

    #We want to compare poisson and gaussian
    binMeans = np.asarray(binMeans)
    binCenters = np.asarray(binCenters)
    
    weightedMean = np.mean((data[0]*10**-6)*10+1)

    #Standard Deviation
    variance = 0
    for i,j in zip(binCenters,binMeans):
        variance += ((i - weightedMean)**2)*j
    
    variance = variance/(numData -1)
    variance = np.sqrt(variance)

    #Poisson and Gaussian values
    poissonValues = []
    gaussianValues = []

    for i in binCenters:
        gaussianValues.append(gaussian(i,weightedMean,variance)*numData)
        poissonValues.append(poisson(weightedMean,i)*numData)
    

    #Test some chi squared values
    #Calculate
    chiSquareExp = reduced_chi_squared_value_calculator(exponential(centers,A,b,c),binMeans,7)
    chiSquareGauss = reduced_chi_squared_value_calculator(gaussianValues,binMeans,7)
    chiSquarePos = reduced_chi_squared_value_calculator(poissonValues,binMeans,8)

    #Different Thresholds
    gaussianThreshhold = (chi2.isf(confidenceInterval,10 -3))/(10 -3)
    poissonThreshhold = chi2.isf(confidenceInterval,10 -2)/(10-2)
    expThreshold = chi2.isf(confidenceInterval,7)

    #Testing the intervals
    print('---------------------------------------')
    print('Number of data points: ' + str(numData))
    #Testing exponential
    if(chiSquareExp < expThreshold):
        print('Exponential fit is consitient')
    elif(chiSquareExp >= expThreshold):
        print('Exponential is not consitent')

    #Testing gaussian
    if(chiSquareGauss < gaussianThreshhold):
        print('Gaussian fit is consistent ')
    elif(chiSquareGauss >= gaussianThreshhold):
        print('Gaussian fit is not consistent')

    #Testing poisson
    if(chiSquarePos < poissonThreshhold):
        print('Poisson fit is consistent ')
    elif(chiSquarePos >= poissonThreshhold):
        print('Poisson fit is not consistent')




    return

#Nums = number of values added in each distributio n
nums = np.linspace(5,100)
for j in range(len(nums)):
    passedData = []

    #Adding the data to the array to be passed
    for i in range(20):
        value = int(nums[j])
        passedData.append(data[i][0:value])
    value = int(nums[j])

    dataChiSquareTester(passedData,value)



