#Importing some fun modules
import numpy as np
import matplotlib.pyplot as plt 

#Plotting
#===================================================================================
# SECTION 4.3
# ALWAYS ON INTEGRATION VS. INTEGRATION IN BAND
#===================================================================================


def fig4_3():

    data = np.loadtxt(r'C:\Users\Andrew\Desktop\Classes\Phys 339\Lab5\4.3_alwaysint.csv', delimiter = ',')
    data = data.transpose()
    plt.errorbar(data[4],data[3], yerr = abs(data[2]), color = 'red', label = 'Integration Always Active')
  
    data = np.loadtxt(r'C:\Users\Andrew\Desktop\Classes\Phys 339\Lab5\4.3_n50ti20.csv', delimiter = ',')
    data = data.transpose()
    data[4] = data[4] + 30
    plt.errorbar(data[4],data[3], yerr = abs(data[2]), label = 'Inter-Band Integration')

    #Limits of the graphs
    plt.xlim(300, 1500)
    plt.ylim(340)
    #Our band lines
    plt.axhline(351,color = 'red', linestyle = '-.', label = 'Upper Band')
    plt.axhline(349,color = 'red', linestyle = '-.', label = 'Lower Band')
    plt.axhline(350, color = 'black', zorder = 100, linestyle = '--', label = 'Set Temp.')
    

    #Background color
    for i in range (349, 351):
        plt.axhspan(i, i+1, facecolor='0.2', alpha=0.1)

    #Title and Axis 
    plt.title('Always Active Integration vs. Proportional Band Integration',fontsize = 14)
    
    plt.ylabel('Temperature [K]', fontsize = 13)
    plt.xlabel('Time Step ~ [0.1s]', fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
        #The legend
    plt.legend() 
    plt.show()      #Showing the graph


#================================================================================
#SECTION 4.4
#PROPORTIONAL INTEGRAL DERIVATIVE CONTROL 
#================================================================================


#================================================================================
#ALL THE DIFFERENT TYPES OF TEMPERATURE CONTROL
#================================================================================
#On off control data
def allTempsPlt():
    #Axis lines
    #Our band lines
    plt.axhline(351,color = 'red', linestyle = '-.', label = 'Upper Band')
    plt.axhline(349,color = 'red', linestyle = '-.', label = 'Lower Band')
    plt.axhline(350, color = 'black', zorder = 100, linestyle = '--', label = 'Set Temp.')
    #Background color
    for i in range (349, 351):
        plt.axhspan(i, i+1, facecolor='0.2', alpha=0.1)

    #Limits
    plt.xlim(40, 1000)
    plt.ylim(345,355)
    #On off
    data = np.loadtxt('4.1_tset-hitting.csv', delimiter = ',')
    data = data.transpose()
    plt.errorbar(data[4],data[3],yerr = data[2], label = 'On Off Control')
    #Proportional
    data = np.loadtxt('4.2_try6.csv',delimiter = ',')
    data = data.transpose()
    plt.errorbar(data[4]- 225,data[3] + 10,yerr = data[2], label = 'Proportional Control')
    #Proportional Integral
    data = np.loadtxt(r'C:\Users\Andrew\Desktop\Classes\Phys 339\Lab5\4.3_n50ti20.csv', delimiter = ',')
    data = data.transpose()
    
    plt.errorbar(data[4]-160,data[3], yerr = abs(data[2]), label = 'Proportional Integration')

    #Show the plot
    plt.title('Temperature vs. Time, Different Control Methods', fontsize = 14)
    plt.ylabel('Temperature [K]', fontsize = 13)
    plt.xlabel('Time Step ~ [0.1s]', fontsize = 13)
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.legend()
    plt.show()
#Proportional Integral Control


#For stability values
data = np.loadtxt(r'C:\Users\Andrew\Desktop\Classes\Phys 339\Lab5\4.3_n50ti20.csv', delimiter = ',').transpose()

#using this we found the data points that we want 
#plt.plot(data[4][700:],data[3][700:])
#plt.show()
#Mean and standard deviation
blueProportionalControlMean = np.mean(data[3][700:])
blueProportionalControlStd = np.std(data[3][700:])

print(blueProportionalControlMean)
print(blueProportionalControlStd)


#Red Proportional 

data = np.loadtxt(r'C:\Users\Andrew\Desktop\Classes\Phys 339\Lab5\4.3_alwaysint.csv', delimiter = ',')
data = data.transpose()
#plt.errorbar(data[4][817:],data[3][817:], yerr = abs(data[2][817:]), color = 'red', label = 'Integration Always Active')
#plt.show()

blueProportionalControlMean = np.mean(data[3][700:])
blueProportionalControlStd = np.std(data[3][700:])

print(blueProportionalControlMean)
print(blueProportionalControlStd)



#Main
'''
fig4_3()
allTempsPlt()
'''