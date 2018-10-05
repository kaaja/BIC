
#%%
#!/usr/bin/env Python3
'''
    This file will read in data and start your mlp network.
    You can leave this file mostly untouched and do your
    mlp implementation in mlp.py.
'''

'''
Mine commments:
    Read in all data
    Last column is the target value
    Make a matrix of target vectors. 
    Target vetcot length = number different target values
    Target vector values =0 for all excpt for the position corresponding to
    the target value.
    Split data into training, validation, test. 50/25/25
    Rescaling by subtracting mean and dividing by max
'''
# Feel free to use numpy in your MLP if you like to.
import numpy as np
#import mlp

filename = 'movements_day1-3.dat'
#filename = 'movements_day1-3Test.dat'


movements = np.loadtxt(filename,delimiter='\t')

# Subtract arithmetic mean for each sensor. We only care about how it varies:
movements[:,:40] = movements[:,:40] - movements[:,:40].mean(axis=0)

# Find maximum absolute value:
imax = np.concatenate(  ( movements.max(axis=0) * np.ones((1,41)) ,
                          np.abs( movements.min(axis=0) * np.ones((1,41)) ) ),
                          axis=0 ).max(axis=0)

# Divide by imax, values should now be between -1,1
movements[:,:40] = movements[:,:40]/imax[:40]
#Me np.shape(movements[:,:40]) #--> (447, 40). One column less than original.

#Me print(np.shape(movements)[0]) #--> 447. 

# Generate target vectors for all inputs 2 -> [0,1,0,0,0,0,0,0] 
# Me: So instead of a scalar between 0 and 7, an array is used.
target = np.zeros((np.shape(movements)[0],8));
for x in range(1,9):
    indices = np.where(movements[:,40]==x)
    target[indices,x-1] = 1
# Me Col 40 is the scalar of the target. 
# Me Fill up all target arrays

# Randomly order the data
order = list(range(np.shape(movements)[0]))
np.random.shuffle(order)
movements = movements[order,:]
target = target[order,:]

# Split data into 3 sets

# Training updates the weights of the network and thus improves the network
train = movements[::2,0:40]
train_targets = target[::2] #Me every 2nd row, start from row0


# Validation checks how well the network is performing and when to stop
valid = movements[1::4,0:40]
valid_targets = target[1::4] #me every 4th row, start row1

# Test data is used to evaluate how good the completely trained network is.
test = movements[3::4,0:40]
test_targets = target[3::4] #me every 4th row, start row3

# My own network

import m2
from m2 import *
numberOfHiddenNodes=9
activationFunction= 'sigmoid'#'linear'
tstRun3 = NN(inputMatrixTrain = train, 
             targetMatrixTrain = train_targets, 
             inputMatrixValid = valid, 
             targetMatrixValid = valid_targets, 
             numberOfHiddenNodes = numberOfHiddenNodes, 
             test = False, 
             activationFunction = activationFunction)
trainingCyclesPerValidation = 15
maxValidations= 300 
maxLocalOptima = 15

tstRun3.solAlg1( trainingCyclesPerValidation = trainingCyclesPerValidation, \
                 maxValidations = maxValidations, maxLocalOptima =maxLocalOptima)
print('tstRun3.validationErrors', tstRun3.validationErrors)

print('tstRun3.totalNumberOfIterations ', tstRun3.totalNumberOfIterations )
tstRun3.predict(test[40,:])
#print('predict: ', tstRun3.zOutput)
print('predict: ', tstRun3.outputPredicted)
print('target: ', test_targets[40,:])


'''
# Try networks with different number of hidden nodes:
hidden = 12


# Initialize the network:
net = mlp.mlp(train, train_targets, hidden)

# Run training:
net.earlystopping(train, train_targets, valid, valid_targets)
# NOTE: You can also call train method from here,
#       and make train use earlystopping method.
#       This is a matter of preference.

# Check how well the network performed:
net.confusion(test,test_targets)
'''
