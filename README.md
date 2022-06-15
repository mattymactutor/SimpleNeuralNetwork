# SimpleNeuralNetwork
C++ : This is a simple neural network used for small scale machine learning. This can be used on a microcontroller.

# Details
This "NeuralNetwork" will use a ".train" file with training data to compute the output weights for a model. The output model is stored in a ".mc" file. This file can be loaded by any other program to use the trained model.

train.format will tell you how to setup the training data to create a model.

EXAMPLES: 
7seg : conversion from bits on 7 segment display to binary numbers.
RGB: predict red green or blue based on pixel value (incredibly trivial)


Output classes must be in binary format.
0 0    0    RED
0 1    1    GREEN
1 0    2    BLUE

The examples provided show how to train a model to classify the 
