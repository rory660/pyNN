import numpy as np
import math

# x        = input
# y        = desired output
# w(n)     = weights for synapse layer n
# z(n)     = neuron values calculated from inputs and weights for layer n
# a(n)     = neuron values after normalisation on layer n
# yHat     = output value, normalised value of final z
# j        = cost function of yHat vs y
# delta(n) = gradient at layer n
# djdw(n)  = gradient applied to a(n+1)?
# s        = scalar value to determine weight change step


class NeuralNetwork:

	def __init__(self):
		self.inputLayerSize = 1
		self.outputLayerSize = 1
		self.hiddenLayerSize = [5,5]

		self.w1 = np.random.rand(self.inputLayerSize, self.hiddenLayerSize[0])
		self.w2 = np.random.rand(self.hiddenLayerSize[0], self.hiddenLayerSize[1])
		self.w3 = np.random.rand(self.hiddenLayerSize[1], self.outputLayerSize)

	def forward(self, x):
		self.z2 =np.dot(x,self.w1)
		self.a2 = self.sigmoid(self.z2)
		self.z3 = np.dot(self.a2,self.w2)
		self.a3 = self.sigmoid(self.z3)
		self.z4 = np.dot(self.a3,self.w3)
		yHat = self.sigmoid(self.z4)
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
		delta4 = np.multiply(-(y-self.yHat), self.sigmoidPrime(self.z4))
		djdw3 = np.dot(self.a3.T, delta4)

		delta3 = np.dot(delta4, self.w3.T)*self.sigmoidPrime(self.z3)
		djdw2 = np.dot(self.a2.T,delta3)

		delta2 = np.dot(delta3, self.w2.T)*self.sigmoidPrime(self.z2)
		djdw1 = np.dot(x.T,delta2)

		return djdw1, djdw2, djdw3

	def correctWeights(self, x, y):
		djdw1, djdw2, djdw3 = self.costFunctionPrime(x,y)
		s = 10
		self.w1 = self.w1 - s * djdw1
		self.w2 = self.w2 - s * djdw2
		self.w3 = self.w3 - s * djdw3

nn = NeuralNetwork()
running = True
i=0
x = np.array([[7485],[6503],[4719]], dtype = float)
x = x/8000
y = np.array([[7000],[6000],[2000]], dtype = float)
y = y/8000


while running:
	i+=1
	if i > 100:
		print y
		print nn.forward(x)
		if raw_input() == "quit":
			running = False
		else:
			i=0
	nn.correctWeights(x,y)