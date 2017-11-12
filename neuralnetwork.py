import numpy as np
import math

# x        = input
# y        = desired output
# w[n]     = weights for synapse layer n (counting from 0)
# z[n]     = neuron values calculated from inputs and weights for layer n+1
# a[n]     = neuron values after normalisation on layer n+1
# yHat     = output value, normalised value of final z
# j        = cost function of yHat vs y
# delta(n) = gradient at layer n
# djdw(n)  = gradient applied to a(n+1)?
# s        = scalar value to determine weight change step


class NeuralNetwork:

	def __init__(self):
		self.inputLayerSize = 1
		self.outputLayerSize = 1
		self.hiddenLayerSize = [5,5,5]

		self.w = []
		self.w.append(np.random.rand(self.inputLayerSize, self.hiddenLayerSize[0]))
		for i in range(len(self.hiddenLayerSize)-1):
			self.w.append(np.random.rand(self.hiddenLayerSize[i+1], self.hiddenLayerSize[i]))
		self.w.append(np.random.rand(self.hiddenLayerSize[-1], self.outputLayerSize))

		self.z = [float(0)]*(len(self.hiddenLayerSize)+1)
		self.a = [float(0)]*(len(self.hiddenLayerSize))

	def forward(self, x):
		self.z[0] =np.dot(x,self.w[0])
		self.a[0] = self.sigmoid(self.z[0])

		for i in range(1,len(self.a)):
			self.z[i] = np.dot(self.a[i-1],self.w[i])
			self.a[i] = self.sigmoid(self.z[i])

		self.z[-1] = np.dot(self.a[-1],self.w[-1])
		yHat = self.sigmoid(self.z[-1])
		return yHat

	def sigmoid(self, n):
		return 1/(1+np.exp(-n))

	def sigmoidPrime(self, n):
		return np.exp(-n)/((1+np.exp(-n))**2)

	def costFunction(self, x,y):
		self.yHat = self.forward(x)
		j = sum(0.5*(y-self.yHat)**2)
		return j

	def costFunctionPrime(self, x, y):
		self.yHat = self.forward(x)

		djdw = []
		delta = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z[-1]))
		djdw.append(np.dot(self.a[-1].T, delta))

		for i in range(len(self.z)-2,0,-1):
		 	delta = np.dot(delta, self.w[i+1].T)*self.sigmoidPrime(self.z[i])
			djdw.append(np.dot(self.a[i-1].T,delta))

		# delta = np.dot(delta, self.w[2].T)*self.sigmoidPrime(self.z[1])
		# djdw.append(np.dot(self.a[0].T,delta))

		delta = np.dot(delta, self.w[1].T)*self.sigmoidPrime(self.z[0])
		djdw.append(np.dot(x.T,delta))

		return list(reversed(djdw))

	def correctWeights(self, x, y):
		s = 10
		djdwList = self.costFunctionPrime(x,y)
		for i in range(len(djdwList)):
			self.w[i] = self.w[i] - s * djdwList[i]

# Usage Format:
# Inputs are always numpy arrays.
# To calculate an output from an input use forward(x), where x is a numpy array of inputs (num of iterations * num of inputs per iteration)
# To correct the weights use correctWeights(x,y), where x is an array of inputs and y is an array of desired outputs. This increments the weight values in favour of the desired outputs.
# Always use values between 0 and 1.

# Example network:

# nn = NeuralNetwork()
# x = np.array([[7485],[6503],[4719]], dtype = float)
# x = x/8000
# y = np.array([[7000],[6000],[2000]], dtype = float)
# y = y/8000

# i=0
# running = True
# while running:
# 	i+=1
# 	if i > 100:
# 		print y
# 		print nn.forward(x)
# 		if raw_input() == "quit":
# 			running = False
# 		else:
# 			i=0
# 	nn.correctWeights(x,y)