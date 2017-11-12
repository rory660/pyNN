import pyNN
import numpy as np

nn = pyNN.NeuralNetwork(1,1,[5,5,5])
x = np.array([[7485],[6503],[4719]], dtype = float)
x = x/8000
y = np.array([[7000],[6000],[2000]], dtype = float)
y = y/8000

i=0
running = True
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
details =  nn.exportDetails()

nn2 = neuralNet.NeuralNetwork()
nn2.importDetails(details)
print nn2.exportDetails()