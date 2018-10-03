

#%%
import numpy as np
import matplotlib.pyplot as plt

class NN:
    """1 hidden layer neutral network. 
    Multidimensional input and output"""
    
    def __init__(self, inputMatrix, targetMatrix, learningRate=.1,\
                 numberOfHiddenNodes=2, maxIterations=1000, method='sequential', \
                 test=False, activationFunction='linear'):
        
        self.inputMatrix, self.targetMatrix, self.learningRate, self.numberOfHiddenNodes,\
                self.maxIterations, self.method, self.test = \
        inputMatrix, targetMatrix, learningRate, numberOfHiddenNodes, \
                 maxIterations, method, test
        
        self.inputMatrixWithBias = np.c_[np.ones(np.shape(inputMatrix)[0]), \
                                    inputMatrix]         
        
        self.wHidden = np.zeros((np.shape(self.inputMatrixWithBias)[0]+1, \
                                            numberOfHiddenNodes))
        self.wOutput = np.zeros((numberOfHiddenNodes+1, \
                                            np.shape(targetMatrix)[1]))
        
        self.hHidden = np.zeros(numberOfHiddenNodes+1)
        self.hHidden[0] = 1
        self.hOutput = np.zeros(np.shape(targetMatrix)[1])
        
        self.zHidden = np.zeros_like(self.hHidden)
        self.zHidden[0] = 1
        self.zOutput = np.zeros_like(self.hOutput)
        
        self.deltaOutput = np.zeros_like(self.zOutput)
        self.deltaHidden = np.zeros_like(self.zHidden)
        
        
        if test:
            self.wHidden = np.array(((1., 1), (-1., 0), (0, 1)))
            self.wOutput  = np.array(((1., 1.), (1., -1.), (0., 1.))) 
            
        if activationFunction == 'linear':
            self.activationFunction = self.activationFunctionLinear
            self.derivative = self.derivativeLinear
        

    def activationFunctionLinear(self, x):
        return x
    
    def derivativeLinear(self, x):
        return 1

    def run(self):
        if self.test:
            numberOfInputVectors = np.shape(self.inputMatrix)[0] -1
        else:
            numberOfInputVectors = np.shape(self.inputMatrix)[0]
       
        for xIndex in range(numberOfInputVectors):
            self.x = self.inputMatrixWithBias[xIndex, :]
            self.targetVector = self.targetMatrix[xIndex, :]
            self.forward()
            self.calculateError()
            self.backward()
                        
    
    def forward(self):
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
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    correct = np.array(((1,0,1), (1,0,1)))
    tstRun = NN(inputMatrix, targetMatrix )
    tstRun.run()
    
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.x - correct ) 
    success = abs(error) < tolerance
    msg = 'abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance)
    assert success, msg
    
def test_init():
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    tstRun = NN(inputMatrix, targetMatrix )

    # Input with bias
    correctMatrix = np.array(((1,0,1), (1,0,1)))
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.inputMatrixWithBias - correctMatrix)
    success = abs(error) < tolerance
    msg = 'Input with bias matrix: abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    # Weight matrix hidden layer
    correctDimensions = np.array((3,2))
    error = np.linalg.norm(np.shape(tstRun.wHidden) - correctDimensions)
    success = abs(error) < tolerance
    msg = 'WeightMatrix hidden abs(error) = %.2E, tolerance = %.2E' \
    %(abs(error), tolerance)
    assert success, msg
    
    # weightMatrixOutput
    correctDimensions = np.array((3,2))
    error = np.linalg.norm(np.shape(tstRun.wOutput) - correctDimensions)
    success = abs(error) < tolerance
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
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    
    tstRun = NN(inputMatrix, targetMatrix, test=True )
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
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    
    tstRun = NN(inputMatrix, targetMatrix, test=True )
    tstRun.run()

    correct = (1.-2)**2 + (0.-2.)**2
    tolerance = 1e-7
    error = correct - tstRun.error2
    success = abs(error) < tolerance
    msg = 'Error : abs(error) = %.2E, tolerance = %.2E' %(abs(error), tolerance),\
    'Error2: ', tstRun.error2
    assert success, msg
    
def test_backward():
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    
    tstRun2 = NN(inputMatrix, targetMatrix, test=True )
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
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    
    tstRun3 = NN(inputMatrix, targetMatrix, test=True )
    tstRun3.x = tstRun3.inputMatrixWithBias[0, :]
    tstRun3.targetVector = tstRun3.targetMatrix[0, :]

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
    




if __name__ == "__main__":
    test_run()
    test_init()   
    test_forward()
    test_backward()
    test_convergence()

    
    