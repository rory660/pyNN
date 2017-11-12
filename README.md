# pyNN
Artifical Neural Network library for Python by Rory Brown  
This was built as a learning exercise.

# Documentation
### This module requires [Python 2.7](https://www.python.org/downloads/) and [NumPy](http://www.numpy.org/) to be used.  
## The NeuralNetwork Class
The NeuralNetwork Class allows for the creation of Neural Network objects, which are input, output and hidden layer configurable.  
### Constructor: NeuralNetwork(int inputSize, int outputSize, list hiddenLayerSizes)  
*inputSize* determines the number of input nodes for the network.  
*outputSize* determines the number of output nodes for the network.  
*hiddenLayerSizes* is a list containing the number of nodes for each hidden layer of the network.  
  
*NeuralNetwork(2,1,[3,3,3])* will generate a NeuralNetwork object with 2 input nodes, 1 output node, and 3 hidden layers with 3 nodes each.  

### forward(numpy.Array x), returns numpy.Array
This method takes an input *x* and uses the network to generate an output *y*.  
*x* is a NumPy Array object that contains input values for the network.  
Each row of the NumPy Array contains one set of input values.  
If the input layer is of size *n*, then each row much contain *n* values. Using multiple rows in the array will allow for multiple outputs to be calculated, which are returned, as *y*, in the same format as input *x*.
