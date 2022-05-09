#ifndef NEURALNETWORK_H_
#define NEURALNETWORK_H_
#include <string>
#include <fstream>
#include <math.h>
using namespace std;

/*
 *TODO
 *
 *
 *
 */

class NeuralNetwork {
private:
	const float LearningRate = 0.3;
	const float Momentum = 0.9;
	const float InitialWeightMax = 0.5;
	const float Success = 0.0004;

	int * RandomizedIndex = nullptr;//[PatternCount];
	long TrainingCycle;
	float Rando;
	float Error;
	float Accum;

	int numInputNodes, numHiddenNodes, numOutputNodes;
	int numTrainingSamples;
	int numClasses;
	string *classNames = nullptr;
	int **trainingData;
	int **trainingClasses;
	float *Hidden = nullptr; //[HiddenNodes];
	float *Output  = nullptr; //[OutputNodes];
	float **HiddenWeights  = nullptr; //[InputNodes + 1][HiddenNodes];
	float **OutputWeights  = nullptr; //[HiddenNodes + 1][OutputNodes];
	float *HiddenDelta  = nullptr; //[HiddenNodes];
	float *OutputDelta = nullptr; //[OutputNodes];
	float **ChangeHiddenWeights = nullptr; //[InputNodes + 1][HiddenNodes];
	float **ChangeOutputWeights = nullptr; //[HiddenNodes + 1][OutputNodes];

	void initHiddenAndChangeWeights();
	void initOutputAndChangeWeights();
	void train();
	void allocate2D( float *** ptr, int rows, int cols);
	void printTrainData();

public:
	NeuralNetwork();
	void trainFromFile(string fileName);
	void saveWeightsToFile(string filename);
	void readWeightsFromFile(string filename);
	string classify(int *data);
	void printCurrentModel();

};

#endif /* NEURALNETWORK_H_ */
