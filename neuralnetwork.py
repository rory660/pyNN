import numpy as np
import math

class NeuralNetwork:

	def __init__(self):
		self.inputLayerSize = 7
		self.outputLayerSize = 1
		self.hiddenLayerSize = 10

		self.w1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize)
		self.w2 = np.random.rand(self.hiddenLayerSize, self.outputLayerSize)

	def forward(self, x):
		self.z2 =np.dot(x,self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.w2)
		yHat = self.sigmoid(self.z3)
		return yHat

	def forwardPrint(self, x):
		print self.sigmoid(self.forward(x))

	def sigmoid(self, n):
		return 1/(1+np.exp(-n))

	def sigmoidPrime(this, n):
		return np.exp(-n)/((1+np.exp(-n))**2)

	def costFunction(this, x):
		self.yHat = self.forward(x)
		j = sum(0.5*(desiredValue-actualValue)**2)
		return j

	def costFunctionPrime(self, x, y):
		self.yHat = self.forward(x)
		delta3 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z3))
		djdw2 = np.dot(self.a2.T, delta3)
		delta2 = np.dot(delta3, self.w2.T)*self.sigmoidPrime(self.z2)
		djdw1 = np.dot(x.T,delta2)

		return djdw1, djdw2

	def correctWeights(self, x, y):
		djdw1, djdw2 = self.costFunctionPrime(x,y)
		s = 0.01
		self.w1 = self.w1 - s* djdw1
		self.w2 = self.w2 - s* djdw2
