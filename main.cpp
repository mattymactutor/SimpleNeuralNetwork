#include <iostream>
using namespace std;

#include "NeuralNetwork.h"


int main() {
	
	NeuralNetwork nn;
	//Run this first to train a model and write an output file with the weights to
	//run it in the future
	//nn.trainFromFile("RGB.train");
	
	//After you have used training to create a .mc file, open that file and 
	//prepare the model
	nn.readWeightsFromFile("7seg.mc");
	
	int data[] = {0,1,1,0,0,0,0}; //9  1001
	cout << "Prediction: " << nn.classify(data);
	return 0;
}


