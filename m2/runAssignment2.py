
#%%

import numpy as np
def dataAssignment2(filename='movements_day1-3.dat'):
	''' Read in all data
		Last column is the target value
		Make a matrix of target vectors. 
		Target vetcot length = number different target values
		Target vector values =0 for all excpt for the position corresponding to
		the target value.
		Split data into training, validation, test. 50/25/25
		Rescaling by subtracting mean and dividing by max
	'''

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

	return train, train_targets, valid, valid_targets, test, test_targets

#train, train_targets, valid, valid_targets, test, test_targets = dataAssignment2()



#import m2
from nnClass import *

def fixedValidationSet(train ,
                       train_targets ,
                       valid ,
                       valid_targets ,
                       numberOfHiddenNodes=9, 
					   activationFunction= 'sigmoid', 
					   trainingCyclesPerValidation = 5, 
					   maxValidations= 300,
					   maxLocalOptima = 15):


	tstRun3 = NN(inputMatrixTrain = train, 
		     targetMatrixTrain = train_targets, 
		     inputMatrixValid = valid, 
		     targetMatrixValid = valid_targets, 
		     numberOfHiddenNodes = numberOfHiddenNodes, 
		     test = False, 
		     activationFunction = activationFunction)
	tstRun3.createWeightsAndLayers()
	tstRun3.solAlg1( trainingCyclesPerValidation = trainingCyclesPerValidation, \
		         maxValidations = maxValidations, maxLocalOptima =maxLocalOptima)

#fixedValidationSet()



# K-fold
def runKfold(	train ,
          	train_targets,
          	valid ,
          	valid_targets ,
			numberOfHiddenNodes=9,
			activationFunction= 'sigmoid',
			numberOfFolds = 3, 
			trainingCyclesPerValidation=1,
			maxValidations = 1000,
			maxLocalOptima = 15,
			printConfusionMatrix=False):



	tstRun4 = NN(inputMatrixTrain = train, 
		         targetMatrixTrain = train_targets, 
		         inputMatrixValid = valid, 
		         targetMatrixValid = valid_targets, 
		         numberOfHiddenNodes = numberOfHiddenNodes, 
		         test = False, 
		         activationFunction = activationFunction)
            
	tstRun4.kFold(numberOfFolds = numberOfFolds,
						trainingCyclesPerValidation = trainingCyclesPerValidation,
						maxValidations=maxValidations,
						maxLocalOptima=maxLocalOptima,
						printConfusionMatrix=printConfusionMatrix)

#runKfold()

# Test set performance
def runTestSet(	train ,
          	train_targets,
          	valid ,
          	valid_targets ,
            test,
            test_targets,
			numberOfHiddenNodes=9,
			activationFunction= 'sigmoid',
			numberOfFolds = 3, 
			trainingCyclesPerValidation=1,
			maxValidations = 1000,
			maxLocalOptima = 15,
			printConfusionMatrix=False, 
            printTestResults=True):

    tstRun3 = NN(inputMatrixTrain = train, 
		     targetMatrixTrain = train_targets, 
		     inputMatrixValid = valid, 
		     targetMatrixValid = valid_targets, 
             inputMatrixTest = test,
		     targetMatrixTest = test_targets,              
		     numberOfHiddenNodes = numberOfHiddenNodes, 
		     test = False, 
		     activationFunction = activationFunction)

    tstRun3.createWeightsAndLayers()
    tstRun3.solAlg1( trainingCyclesPerValidation = trainingCyclesPerValidation, \
		         maxValidations = maxValidations, maxLocalOptima =maxLocalOptima)
    tstRun3.wHidden = tstRun3.wHiddenBest
    tstRun3.wOutput = tstRun3.wOutputBest
    tstRun3.targetMatrixValid = tstRun3.targetMatrixTest
    tstRun3.inputMatrixValid = tstRun3.inputMatrixTest
    tstRun3.targetVector = tstRun3.targetMatrixTest[0, :]
    tstRun3.calculateValidationError(printTestResults=printTestResults)
    #print(tstRun3.confusionMatrix)
	
