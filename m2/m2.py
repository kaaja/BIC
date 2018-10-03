import numpy as np
import matplotlib.pyplot as plt

#%%

class NN:
    """1 hidden layer neutral network. 
    Multidimensional input and output"""
    
    def __init__(self, inputMatrix, targetMatrix, learningRate=.1,\
                 numberOfHiddenNodes=2, maxIterations=1000, method='sequential'):
        
        self.inputMatrix, self.targetMatrix, self.learningRate, numberOfHiddenNodes,\
                self.maxIterations, self.method = \
        inputMatrix, targetMatrix, learningRate, numberOfHiddenNodes, \
                 maxIterations, method
        
        self.inputMatrixWithBias = np.c_[np.ones(np.shape(inputMatrix)[0]), \
                                    inputMatrix]         
        self.wHidden = np.zeros((np.shape(self.inputMatrixWithBias)[0]+1, \
                                            numberOfHiddenNodes))
        
        self.wOutput = np.zeros((numberOfHiddenNodes+1, \
                                            np.shape(targetMatrix)[0]))
        
        self.hHidden = np.zeros(numberOfHiddenNodes+1)
        self.hOutput = np.zeros(np.shape(targetMatrix)[0])
        
        self.zHidden = np.zeros_like(self.hHidden+1)
        self.hOutput = np.zeros_like(self.hOutput)
        

    def run(self):
        for inputVectorIndex in range(np.shape(self.inputMatrix)[0]):
            self.inputVector = self.inputMatrix[inputVectorIndex, :]
            self.targetVector = self.targetMatrix[inputVectorIndex, :]
            
    def forward(self):
        for j in range(1, self.NumberOfHiddenNodes):
            for i in range(len(self.inputVector)):
                a = 1
            
    
         
def test_run():
    inputMatrix = np.array(((0,1), (0,1)))
    targetMatrix = np.array(((1,0), (1,0)))
    tstRun = NN(inputMatrix, targetMatrix )
    tstRun.run()
    
    tolerance = 1e-7
    error = np.linalg.norm(tstRun.inputVector - inputMatrix[1]) 
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


if __name__ == "__main__":
    test_run()
    test_init()    