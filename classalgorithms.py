from __future__ import division  # floating point division
import numpy as np
import utilities as utils
import math
from sklearn.linear_model import LogisticRegression
from  scipy.optimize import minimize as mini
from  sklearn.cluster import KMeans

class Classifier:
    """
    Generic classifier interface; returns random classification
    Assumes y in {0,1}, rather than {-1, 1}
    """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        self.params = {}
        
    def reset(self, parameters):
        """ Reset learner """
        self.resetparams(parameters)

    def resetparams(self, parameters):
        """ Can pass parameters to reset with new parameters """
        try:
            utils.update_dictionary_items(self.params,parameters)
        except AttributeError:
            # Variable self.params does not exist, so not updated
            # Create an empty set of params for future reference
            self.params = {}
            
    def getparams(self):
        return self.params
    
    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        
    def predict(self, Xtest):
        probs = np.random.rand(Xtest.shape[0])
        ytest = utils.threshold_probs(probs)
        return ytest

class LinearRegressionClass(Classifier):
    """
    Linear Regression with ridge regularization
    Simply solves (X.T X/t + lambda eye)^{-1} X.T y/t
    """
    def __init__( self, parameters={} ):
        self.params = {'regwgt': 0.01}
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None

    def learn(self, Xtrain, ytrain):
        """ Learns using the traindata """
        # Ensure ytrain is {-1,1}
        yt = np.copy(ytrain)
        yt[yt == 0] = -1
        
        # Dividing by numsamples before adding ridge regularization
        # for additional stability; this also makes the
        # regularization parameter not dependent on numsamples
        # if want regularization disappear with more samples, must pass
        # such a regularization parameter lambda/t
        numsamples = Xtrain.shape[0]
        self.weights = np.dot(np.dot(np.linalg.pinv(np.add(np.dot(Xtrain.T,Xtrain)/numsamples,self.params['regwgt']*np.identity(Xtrain.shape[1]))), Xtrain.T),yt)/numsamples
        
    def predict(self, Xtest):
        ytest = np.dot(Xtest, self.weights)
        ytest[ytest > 0] = 1     
        ytest[ytest < 0] = 0    
        return ytest
        
class NaiveBayes(Classifier):
    """ Gaussian naive Bayes;  """
    
    def __init__( self, parameters={} ):
        """ Params can contain any useful parameters for the algorithm """
        # Assumes that a bias unit has been added to feature vector as the last feature
        # If usecolumnones is False, it ignores this last feature
        self.params = {'usecolumnones': False}
        self.reset(parameters)
        self.unique_class_counts = {'total_count': 0, 'class_counts': {}}
        self.vector = []
        self.class_count = {}
        self.summaries = {}


    def reset(self, parameters):
        self.Number_Classes = {'total': 0, 'total_classes':{}}

        # TODO: set up required variables for learning

    def mean(numbers):
        return sum(numbers) / float(len(numbers))

    def stdev(numbers):
        avg = NaiveBayes.mean(numbers)
        variance = sum([pow(x - avg, 2) for x in numbers]) / float(len(numbers) - 1)
        return math.sqrt(variance)

    def summarize(self):
        values = [(NaiveBayes.mean(attribute), NaiveBayes.stdev(attribute)) for attribute in zip(*self.vector)]
        del values[-1]
        return values

    def summarize_N(self, Xtrain):
        values = [(NaiveBayes.mean(attribute), NaiveBayes.stdev(attribute)) for attribute in zip(*self.vector)]
        del values[-1]
        return values

    def summarizeByClass(self):

        self.summariesByClassVar = {}
        for classValue, instances in self.class_count.items():
            self.summariesByClassVar[classValue] = NaiveBayes.summarize_N(self, instances)
        return self.summariesByClassVar

    def learn(self, Xtrain, ytrain):
        self.vector = np.c_[Xtrain, ytrain]
        for i in range(len(Xtrain)):

            if (self.vector[i][-1] not in self.class_count):
                self.class_count[self.vector[i][-1]] = []
            self.class_count[self.vector[i][-1]].append(self.vector)
        #NaiveBayes.separateByClass(self, Xtrain)
        NaiveBayes.summarize(self)
        NaiveBayes.summarizeByClass(self)



    def calculateProbability(x, mean, stdev):
        exponent = math.exp(-(math.pow(x - mean, 2) / (2 * math.pow(stdev, 2))))
        return (1 / (math.sqrt(2 * math.pi) * stdev)) * exponent

        #for the calculateClassProbabilities() function


    def calculateClassProbabilities(self, inputVector):
        probabilities = {}
        for classValue, classSummaries in self.summariesByClassVar.items():
            probabilities[classValue] = 1
            for i in range(len(classSummaries)-2):
                mean, stdev = classSummaries[i]
                x = inputVector[i]
                probabilities[classValue] *= NaiveBayes.calculateProbability(x, mean, stdev)
        return probabilities

    def predict_val(self, inputVector):
        probabilities = NaiveBayes.calculateClassProbabilities(self, inputVector)
        bestLabel, bestProb = None, -1
        for classValue, probability in probabilities.items():
            if bestLabel is None or probability > bestProb:
                bestProb = probability
                bestLabel = classValue
        return bestLabel

    def predict(self, Xtest):
        ytest = []
        if not self.params['usecolumnones']:
            Xtest = Xtest[:, :-1]

        self.summaries = self.class_count
        for i in range(len(Xtest)):
            result = NaiveBayes.predict_val(self, Xtest[i])
            ytest.append(result)

        return ytest

class LogitReg(Classifier):

    def __init__( self, parameters={} ):
        # Default: no regularization
        self.params = {'regwgt': 0.0, 'regularizer': 'None'}
        self.reset(parameters)


    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        if self.params['regularizer'] is 'l1':
            self.regularizer = (utils.l1, utils.dl1)
        elif self.params['regularizer'] is 'l2':
            self.regularizer = (utils.l2, utils.dl2)
        else:
            self.regularizer = (lambda w: 0, lambda w: np.zeros(w.shape,))


    def gaussian_kernels(self, Xtrain, h):
        N = len(Xtrain)
        x = np.random.uniform(0.0, 1.0, N)
        x_output = []
        #x = KMeans(n_clusters= 3).fit(Xtrain).cluster_centers_

        for i in np.arange(N):
            norm = (2 * np.pi * h ** 2) ** (-0.5)
            e = np.exp(-(x[i] - Xtrain) ** 2 / (2 * h ** 2))
            x_output.append(sum(norm * e) / N)

        return np.array(x_output)



        # def gaussian_kernel(x, y, sigma=5.0):
    #    return np.exp(-np.linalg.norm(x - y) ** 2 / (2 * (sigma ** 2)))

    def predict_prob(feature_matrix, weights):
        #feature_matrix = Xtrain.shape[0]
        #self.weights = np.zeros(Xtrain.shape[1])
        DotRet = np.dot(feature_matrix, weights)
        result = 1 / (1 + np.exp(-DotRet))
        return  result



    def feature_derivative(features, errors):
        derivatives = np.dot(np.transpose(features), errors)
        return derivatives

    def learn(self, Xtrain, ytrain):

        self.features = Xtrain

        numsamples = Xtrain.shape[0]

      #  Xtrain = LogitReg.gaussian_kernels(self, Xtrain, 0.75) # Function to call Kernel Fun().
        feature_matrix = Xtrain
        self.weights = np.ones((Xtrain.shape[1], 1))
        #print ("ytrain shape is:" + str(ytrain.shape))
        Ytrain = ytrain.reshape((len(ytrain), 1))
        for i in yrange(1000):
            print("hi")
            pred = LogitReg.predict_prob(feature_matrix, self.weights)
            error = Ytrain - pred
            Penalty = 0
            gradient = np.dot(Xtrain.T, error)

            if (self.params['regularizer'] is 'l2'):

                Penalty = LogitReg.log_likelihood(self, Xtrain, ytrain)
            elif (self.params['regularizer'] is 'l1'):
                Penalty = LogitReg.log_likelihood(self,Xtrain, ytrain)

            else:
                 self.weights = self.weights + 0.001*gradient + Penalty


        return self.weights

        #prediction = predict(feature_matrix, self.weights)


    def log_likelihood(self, Xtrain, ytrain):

        scores = np.dot(Xtrain, self.weights)
        logexp = np.log(1 + np.exp(-scores))

        mask = np.isinf(logexp)
        logexp[mask] = -scores[mask]

        likelihood = np.sum((1 - ytrain) * scores + logexp)
        likelihood = likelihood + np.dot(0.01 / (2 * Xtrain.size), np.sum(self.weights * self.weights))
        if (self.params['regularizer'] is 'l2'):
            likelihood = likelihood + np.dot(0.01 / (2 * Xtrain.size), np.sum(np.abs(self.weights)))
        elif (self.params['regularizer'] is 'l1'):
            likelihood = likelihood + np.dot(0.01 / (2 * Xtrain.size), np.sum(self.weights * self.weights))

        return likelihood

    def predict(self, Xtest ):


        DotRet = np.dot(Xtest, self.weights)
        ytest = result = 1 / (1 + np.exp(-DotRet))
        ytest[ytest >= .5] = 1
        ytest[ytest < .5] = 0
        return ytest




class NeuralNet(Classifier):

    def __init__(self, parameters={}):
        self.params = {'nh': 4,
                        'transfer': 'sigmoid',
                        'stepsize': 0.01,
                        'epochs': 10}
        self.reset(parameters)        

    def reset(self, parameters):
        self.resetparams(parameters)
        if self.params['transfer'] is 'sigmoid':
            self.transfer = utils.sigmoid
            self.dtransfer = utils.dsigmoid
        else:
            # For now, only allowing sigmoid transfer
            raise Exception('NeuralNet -> can only handle sigmoid transfer, must set option transfer to string sigmoid')      
        self.wi = None
        self.wo = None

    def sigmoid(self, z):
        # Apply sigmoid activation function to scalar, vector, or matrix
        return 1 / (1 + np.exp(-z))

    def forward(self, Xtrain):
        self.wi = np.ones(Xtrain.T.shape)
        self.w0 = np.ones(Xtrain.shape[0])
        self.Z2 = np.dot(Xtrain, self.wi)
        self.activation = self.sigmoid(self.Z2)
        self.Z3 = np.dot(self.activation, self.w0)
        result = self.sigmoid(self.Z3)
        return result

    #Sigmoid for Gradient decent
    def sigmodiPrime(self, z):
        return np.exp(-z)/((1 + np.exp(-z))**2)


    def learn(self, Xtrain, ytrain):

        self.wi = np.ones(Xtrain.T.shape)
        self.w0 = np.ones(Xtrain.shape[0])
        self.Z2 = np.dot(Xtrain, self.wi)
        self.activation = self.sigmoid(self.Z2)
        self.Z3 = np.dot(self.activation, self.w0)
        self.result = self.sigmoid(self.Z3)
        for  i in range(1000):

            delta3 = np.multiply(-(ytrain - self.result), NeuralNet.sigmodiPrime(self, self.Z3))
        # Add gradient of regularization term:
            dJdW2 = np.dot(self.activation.T, delta3) / Xtrain.shape[0] + 0.01 * self.w0

            delta2 = np.dot(delta3, self.w0.T) * NeuralNet.sigmodiPrime(self, self.Z2)
        # Add gradient of regularization term:
            dJdW1 = np.dot(Xtrain.T, delta2) / Xtrain.shape[0] + 0.01 * self.wi

            self.w0 += 0.01*dJdW2
            self.wi += 0.01*dJdW1
            return self.w0, self.wi

    def costFunctionPrime(self, Xtrain, ytrain):
        # Compute derivative with respect to W and W2 for a given X and y:
        self.result = self.forward(Xtrain)

        delta3 = np.multiply(-(ytrain - self.result), NeuralNet.sigmodiPrime(self, self.Z3))
        # Add gradient of regularization term:
        dJdW2 = np.dot(self.activation.T, delta3) / Xtrain.shape[0] + 0.01 * self.w0

        delta2 = np.dot(delta3, self.w0.T) * NeuralNet.sigmodiPrime(self, self.Z2)
        # Add gradient of regularization term:
        dJdW1 = np.dot(Xtrain.T, delta2) / Xtrain.shape[0] + 0.01 * self.wi

        return dJdW1, dJdW2

    def computeGradients(self, Xtrain, ytrain):
        dJdW1, dJdW2 = self.costFunctionPrime(Xtrain, ytrain)
        return np.concatenate((dJdW1.ravel(), dJdW2.ravel()))

    def predict(self, Xtest):
        #ytest = self.sigmoid(Xtest , self.wi)
        self.Z2 = np.dot(Xtest, self.wi)
        self.activation = self.sigmoid(self.Z2)
        self.Z3 = np.dot(self.activation, self.w0)
        ytest = self.sigmoid(self.Z3)
        ytest[ytest >= .5] = 1
        ytest[ytest < .5] = 0
        return ytest
    
    def _evaluate(self, inputs):
        """ 
        Returns the output of the current neural network for the given input
        The underscore indicates that this is a private function to the class NeuralNet
        """
        if inputs.shape[0] != self.ni:
            raise ValueError('NeuralNet:evaluate -> Wrong number of inputs')
        
        # hidden activations
        ah = self.transfer(np.dot(self.wi,inputs))  

        # output activations
        ao = self.transfer(np.dot(self.wo,ah))
        
        return (ah, ao)


class LogitRegAlternative(Classifier):

    def __init__( self, parameters={} ):
        self.reset(parameters)

    def reset(self, parameters):
        self.resetparams(parameters)
        self.weights = None
        
    # TODO: implement learn and predict functions                  
           
    
