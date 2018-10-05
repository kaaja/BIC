#%%


import numpy as np
import matplotlib.pyplot as plt

class NN:
    """1 hidden layer neutral network. 
    Multidimensional input and output"""
    
    def __init__(self, inputMatrixTrain, targetMatrixTrain, 
                 inputMatrixValid=False, targetMatrixValid=False, \
                 inputMatrixTest=False, targetMatrixTest=False, learningRate=.1,\
                 numberOfHiddenNodes=2, maxIterations=1000, method='sequential', \
                 test=False, activationFunction='linear'):
        
        self.inputMatrixTrain = inputMatrixTrain
        self.targetMatrixTrain = targetMatrixTrain
        self.inputMatrixValid = inputMatrixValid
        self.targetMatrixValid = targetMatrixValid
        self.inputMatrixTest = inputMatrixTest
        self.targetMatrixTest = targetMatrixTest
        self.learningRate = learningRate
        self.numberOfHiddenNodes =  numberOfHiddenNodes
        self.maxIterations = maxIterations
        self.method = method
        self.test = test\
         
        
        self.inputMatrixTrainWithBias = np.c_[np.ones(np.shape(inputMatrixTrain)[0]), \
                                    inputMatrixTrain]         
        
        
        self.hHidden = np.zeros(numberOfHiddenNodes+1)
        self.hHidden[0] = 1
        self.hOutput = np.zeros(np.shape(targetMatrixTrain)[1])
        
        self.zHidden = np.zeros_like(self.hHidden)
        self.zHidden[0] = 1
        self.zOutput = np.zeros_like(self.hOutput)
        
        self.deltaOutput = np.zeros_like(self.zOutput)
        self.deltaHidden = np.zeros_like(self.zHidden)
        
        
        if test:
            self.wHidden = np.array(((1., 1), (-1., 0), (0, 1)))
            self.wOutput  = np.array(((1., 1.), (1., -1.), (0., 1.))) 
         
        else:
            self.wHidden = np.random.random_sample((np.shape(self.inputMatrixTrainWithBias)[1],\
                                            self.numberOfHiddenNodes)) - .5
            self.wOutput = np.random.random_sample((self.numberOfHiddenNodes+1,\
                                            np.shape(self.targetMatrixTrain)[1])) - .5
            
        if activationFunction == 'linear':
            self.activationFunction = self.activationFunctionLinear
            self.derivative = self.derivativeLinear
        elif activationFunction == 'sigmoid':
            self.activationFunction = self.sigmoid
            self.derivative = self.derivativeSigmoid
        

    def activationFunctionLinear(self, x):
        return x
    
    def derivativeLinear(self, x):
        return 1
    
    def sigmoid(self, x):
        return 1./(1.+np.exp(-x))
                
    def derivativeSigmoid(self, x):
        return self.sigmoid(x)*(1. - self.sigmoid(x))
    
    def solAlg1(self, 
                trainingCyclesPerValidation=100, 
                maxIterationsEarlyStopping= 5000):
        """
        Fixed validation set.
        Weight change for every input.
        Random order of inputs"""
        
        self.validationErrors = []
        valErrorNew = 5e10
        valErrorOld = valErrorNew + 1.
        localOptimas = 0
        while valErrorNew < valErrorOld:
            for trainingCycle in range(trainingCyclesPerValidation):
                #print('trainingCycle ', trainingCycle )
                indices = list(range(np.shape(self.inputMatrixTrainWithBias)[0]))
                #print('len(indices)',len(indices))
                np.random.shuffle(indices)
                
                self.inputMatrixTrainWithBias = \
                self.inputMatrixTrainWithBias[indices,:]
                
                self.targetMatrixTrain= \
                self.targetMatrixTrain[indices,:]
                
                for idx in range(np.shape(self.targetMatrixTrain)[0]):
                    self.x = self.inputMatrixTrainWithBias[idx, :]
                    self.targetVector = self.targetMatrixTrain[idx, :]
                    self.forward()
                    self.backward()   
            self.calculateValidationError()
            valErrorOld = valErrorNew
            valErrorNew = self.valError
            self.validationErrors.append(valErrorNew)
            #print(valErrorNew)
        #print(self.validationErrors)
            
                
    def calculateValidationError(self):
        self.valError = 0
        predictionMatrixValid = np.zeros_like(self.targetMatrixValid)
        
        confusionMatrix = np.zeros((len(self.targetVector), \
                                    len(self.targetVector)))

        for idx in range(np.shape(self.inputMatrixValid)[0]):
            #self.x = self.inputMatrixValid[idx, :]
            x = self.inputMatrixValid[idx, :]
            #print('x', x)
            #self.targetVector = self.targetMatrixValid[idx, :]
            targetVector = self.targetMatrixValid[idx, :]
            #self.forward()
            prediction = self.forwardNonTrain(x)
            #print('prediction', prediction)
            #self.hardMax(self.zOutput)
            self.hardMax(prediction)
            predictionMatrixValid[idx,:] = self.hardMaxValue
            #print('predictionMatrixValid[idx,:]', predictionMatrixValid[idx,:])
            
            confusionMatrix[np.argmax(targetVector), \
                            np.argmax(self.hardMaxValue)] +=1
            
            
            '''
            #for outputComponent in range(len(self.targetVector)):
            for outputComponent in range(len(targetVector)):
                #print('self.zOutput',self.zOutput)
                #self.valError += (self.hardMaxValue[outputComponent] - \
                 #            self.targetVector[outputComponent])**2
                self.valError += (self.hardMaxValue[outputComponent] - \
                             targetVector[outputComponent])**2
            '''
        maxIndexValid = np.argmax(self.targetMatrixValid, axis=1)
        maxIndexPrediction = np.argmax(predictionMatrixValid, axis=1)
        #print(self.targetMatrixValid)
        #print('maxIndexValid', maxIndexValid)
        #print(predictionMatrixValid)
        #print('maxIndexPrediction', maxIndexPrediction)
        accuracy = np.sum(maxIndexValid == maxIndexPrediction)/len(maxIndexValid)
        self.valError= 1. - accuracy 
        
        accuracyConfusionMatrix = float(np.trace(confusionMatrix)/np.sum(confusionMatrix))
        #print('accuracyConfusionMatrix/accuracy', accuracyConfusionMatrix/accuracy)
        #print('confusionMatrix \n' , confusionMatrix)
        #print('accuracyConfusionMatrix ', accuracyConfusionMatrix )
        #print('accuracy',accuracy)
        #print(self.valError)
        
            
    def predict(self, x):
        #self.x = x
        #self.forward()
        self.outputPredicted = self.forwardNonTrain(x)
        
    def hardMax(self, x):
        self.hardMaxValue = np.copy(x)
        self.hardMaxValue[np.argmax(self.hardMaxValue)] = 1
        self.hardMaxValue = np.round(self.hardMaxValue)

    def run(self):
        if self.test:
            numberOfInputVectors = np.shape(self.inputMatrixTrain)[0] -1
        else:
            numberOfInputVectors = np.shape(self.inputMatrixTrain)[0]
       
        for xIndex in range(numberOfInputVectors):
            self.x = self.inputMatrixTrainWithBias[xIndex, :]
            self.targetVector = self.targetMatrixTrain[xIndex, :]
            self.forward()
            self.calculateError()
            self.backward()
                        
    
    
    
    def forward(self):
        #Training
        for j in range(1, self.numberOfHiddenNodes+1):
            self.hHidden[j] = 0
            for i in range(len(self.x)):
                self.hHidden[j] += self.wHidden[i,j-1]*self.x[i]
            self.zHidden[j] = self.activationFunction(self.hHidden[j])
            
        for k in range(len(self.hOutput)):
            self.hOutput[k] = 0
            for j in range(len(self.hHidden)):
                self.hOutput[k] += self.wOutput[j,k]*self.zHidden[j]
            self.zOutput[k] = self.activationFunction(self.hOutput[k])
            
    def forwardNonTrain(self, x):
        hHidden = np.zeros_like(self.hHidden)
        zHidden = np.zeros_like(self.zHidden)
        for j in range(1, self.numberOfHiddenNodes+1):
            hHidden[j] = 0
            for i in range(len(x)):
                hHidden[j] += self.wHidden[i,j-1]*x[i]
            zHidden[j] = self.activationFunction(hHidden[j])
        
        hOutput = np.zeros_like(self.hOutput)
        zOutput = np.zeros_like(self.zOutput)
        for k in range(len(hOutput)):
            hOutput[k] = 0
            for j in range(len(hHidden)):
                hOutput[k] += self.wOutput[j,k]*zHidden[j]
            zOutput[k] = self.activationFunction(hOutput[k])
        return zOutput
                
    def calculateError(self):
        self.error2 = 0
        for k in range(len(self.zOutput)):
            self.error2 += (self.zOutput[k] - self.targetVector[k])**2
            
    def backward(self):
        for k in range(len(self.zOutput)):
            self.deltaOutput[k] = (self.zOutput[k] - self.targetVector[k])\
            *self.derivative(self.hOutput[k])
            
        for j in range(len(self.hHidden)):
            self.deltaHidden[j] = 0
            for k in range(len(self.zOutput)):
                self.deltaHidden[j] += self.deltaOutput[k]*self.wOutput[j,k]
                
        for j in range(np.shape(self.wOutput)[0]):
            for k in range(np.shape(self.wOutput)[1]):
                self.wOutput[j,k] -= self.learningRate*self.deltaOutput[k]\
                *self.zHidden[j]
        for i in range(np.shape(self.wHidden)[0]):
            for j in range(np.shape(self.wHidden)[1]):
                self.wHidden[i,j] -= self.learningRate*self.deltaHidden[j+1]\
                *self.x[i]
            
        
    
         
def test_run():
    inputMatrixTrain = np.array(((0,1), (0,1)))
    targetMatrixTrain = np.array(((1,0), (1,0)))
    correct = np.array(((1,0,1), (1,0,1)))
    tstRun = NN(inputMatrixTrain, targetMatrixTrain )
    tstRun.run()
    
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.x - correct ) 
    success = abs(error) < tolerance
    msg = 'abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance)
    assert success, msg
    
def test_init():
    inputMatrixTrain = np.array(((0,1), (0,1)))
    targetMatrixTrain = np.array(((1,0), (1,0)))
    tstRun = NN(inputMatrixTrain, targetMatrixTrain )

    # Input with bias
    correctMatrix = np.array(((1,0,1), (1,0,1)))
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.inputMatrixTrainWithBias - correctMatrix)
    success = abs(error) < tolerance
    msg = 'Input with bias matrix: abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    # Weight matrix hidden layer
    correctDimensions = np.array((3,2))
    error = np.linalg.norm(np.shape(tstRun.wHidden) - correctDimensions)
    error2 = np.linalg.norm(np.zeros((3,2)) - tstRun.wHidden) # Not zeros only
    success = abs(error) < tolerance and error2 > 0
    msg = 'WeightMatrix hidden abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    # weightMatrixOutput
    correctDimensions = np.array((3,2))
    error = np.linalg.norm(np.shape(tstRun.wOutput) - correctDimensions)
    error2 = np.linalg.norm(np.zeros((3,2)) - tstRun.wOutput) # Not zeros only
    success = abs(error) < tolerance and error2 > 0
    msg = 'wOutput abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    # hHidden
    correct = np.zeros(3)
    correct[0] = 1
    error = np.linalg.norm(tstRun.hHidden - correct)
    success = abs(error) < tolerance
    msg = 'hHidden abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    # hOutput
    correct = np.zeros(2)
    error = np.linalg.norm(tstRun.hOutput - correct)
    success = abs(error) < tolerance
    msg = 'hOutput abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    
    
def test_forward():
    inputMatrixTrain = np.array(((0,1), (0,1)))
    targetMatrixTrain = np.array(((1,0), (1,0)))
    
    tstRun = NN(inputMatrixTrain, targetMatrixTrain, test=True )
    tstRun.run()
    
    # hHidden
    hHiddenCorrect = np.array((1,1,2))
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.hHidden - hHiddenCorrect) 
    success = abs(error) < tolerance
    msg = 'forward, hHidden: abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'hHidden numerical: ', tstRun.hHidden
    assert success, msg
    
    # zHidden
    zHiddenCorrect = np.array((1,1,2))
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.zHidden - zHiddenCorrect)  
    success = abs(error) < tolerance
    msg = 'forward, zHidden: abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'hHidden numerical: ', tstRun.zHidden
    assert success, msg
    
    # hOutput
    hOutputCorrect = np.array((2,2))
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.hOutput - hOutputCorrect ) 
    success = abs(error) < tolerance
    msg = 'forward, hOutput : abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'hOutput numerical: ', tstRun.hOutput 
    assert success, msg
    
    # zOutput
    zOutputCorrect = np.array((2,2))
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.zOutput - zOutputCorrect ) 
    success = abs(error) < tolerance
    msg = 'forward, hOutput : abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'hOutput numerical: ', tstRun.zOutput 
    assert success, msg

def test_calculateError():
    inputMatrixTrain = np.array(((0,1), (0,1)))
    targetMatrixTrain = np.array(((1,0), (1,0)))
    
    tstRun = NN(inputMatrixTrain, targetMatrixTrain, test=True )
    tstRun.run()

    correct = (1.-2)**2 + (0.-2.)**2
    tolerance = 1e-7
    error = correct - tstRun.error2
    success = abs(error) < tolerance
    msg = 'Error : abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'Error2: ', tstRun.error2
    assert success, msg
    
def test_backward():
    inputMatrixTrain = np.array(((0,1), (0,1)))
    targetMatrixTrain = np.array(((1,0), (1,0)))
    
    tstRun2 = NN(inputMatrixTrain, targetMatrixTrain, test=True )
    tstRun2.run()

    # deltaOutput
    correct = np.array((1, 2))
    tolerance = 1e-7
    error = np.linalg.norm(correct - tstRun2.deltaOutput)
    success = abs(error) < tolerance
    msg = 'deltaOutput : abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'deltaOutput: ', tstRun2.deltaOutput
    assert success, msg
    
    # deltaHidden
    correct = np.array((-1, 2))
    tolerance = 1e-7
    error = np.linalg.norm(correct - tstRun2.deltaHidden[1:]) 
    success = abs(error) < tolerance
    msg = 'deltaHidden: abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'deltaHidden: ', tstRun2.deltaHidden
    assert success, msg
    
    # wOutput
    correct = np.array(((0.9, -1.2), (-0.2, 0.6)))
    tolerance = 1e-7
    error = np.linalg.norm(correct - tstRun2.wOutput[1:,:]) 
    success = abs(error) < tolerance
    msg = 'wOutput: abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'wOutput: ', tstRun2.wOutput[1:,:]
    assert success, msg
    
    # wHidden
    correct = np.array(((-1., 0.), (0.1, 0.8)))
    tolerance = 1e-7
    error = np.linalg.norm(correct - tstRun2.wHidden[1:,:])
    success = abs(error) < tolerance
    msg = 'wHidden: abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'wHidden: ', tstRun2.wHidden[1:,:]
    assert success, msg
    
def test_convergence():
    inputMatrixTrain = np.array(((0,1), (0,1)))
    targetMatrixTrain = np.array(((1,0), (1,0)))
    
    tstRun3 = NN(inputMatrixTrain, targetMatrixTrain, test=True )
    tstRun3.x = tstRun3.inputMatrixTrainWithBias[0, :]
    tstRun3.targetVector = tstRun3.targetMatrixTrain[0, :]

    tolerance = 1e-4
    tstRun3.error2 = 5
    iteration = 0
    
    while abs(tstRun3.error2) > tolerance and iteration < tstRun3.maxIterations:
        tstRun3.forward()
        tstRun3.calculateError()
        tstRun3.backward()
        iteration += 1
    success = abs(tstRun3.error2) < tolerance
    msg = 'Convergence: abs(error) = %.2E, tolerance = %.2E' %(abs(tstRun3.error2), tolerance),\
    'Iterations: ', iteration, 'Output: ', tstRun3.zOutput
    assert success, msg
    
def test_solAlg1():
    inputMatrixTrain = np.array(((0,1), (1,0)))
    targetMatrixTrain = np.array(((1,0), (0,1)))
    inputMatrixValid = np.array(((1,1), (0, 0)))
    targetMatrixValid = np.array(((0, 0), (1,1))) 
    
    tstRun3 = NN(inputMatrixTrain, targetMatrixTrain, 
                 inputMatrixValid , targetMatrixValid , test=True )
    tstRun3.solAlg1()
    
    # Early stopping
    validationErrorsArray = np.array(tstRun3.validationErrors)
    validationDirection = np.zeros(len(validationErrorsArray)-1)
    for i in range(len(validationDirection)):
        validationDirection[i] = validationErrorsArray[i+1]\
        -validationErrorsArray[i]
    wrong1 = 0
    wrong2 = 0
    for i in range(len(validationDirection)):
        if i < len(validationDirection) -1 and validationDirection[i] > 0:
            #print('validationDirection[i]',validationDirection[i])
            wrong1 = 1
    if validationDirection[-1] < 0:
            wrong2 = 1
    success = (wrong1+ wrong2) == 0
    #success = abs(tstRun3.error2) < tolerance
    msg = 'Early stopping array change validation error: ',  validationDirection, \
    'Wrong1, wrong 2', wrong1, wrong2
    assert success, msg





if __name__ == "__main__":
    test_run()
    test_init()   
    test_forward()
    test_backward()
    test_convergence()
    test_solAlg1()
    
#%%