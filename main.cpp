#include <math.h>
#include <time.h>       /* time */
#include <iostream>
#include <fstream>
using namespace std;

//http://robotics.hobbizine.com/arduinoann.html

/*
 * This program can train a model for simple truth table
 * expressions. This example converts the light outputs for a 7 segment display
 * to the actual number being shown. Once the model has been trained, the weights
 * and node information is saved to a .mc file. At a later time, the .mc file
 * can be loaded and a pre-trained model can be used to make a classification
 *
 *
 * Next Steps - make a model that can classify a pixel as rgb
 *
 */

/******************************************************************
 * Network Configuration - customized per network
 ******************************************************************/

const int PatternCount = 10;
const int InputNodes = 7;
const int HiddenNodes = 8;
const int OutputNodes = 4;
const float LearningRate = 0.3;
const float Momentum = 0.9;
const float InitialWeightMax = 0.5;
const float Success = 0.0004;

int numInputNodes, numHiddenNodes, numOutputNodes;
int numTrainingSamples;
int numClasses;
string *classNames;
int **trainingData;
int **trainingClasses;

const int Input[PatternCount][InputNodes] = {
		{ 1, 1, 1, 1, 1, 1, 0 },  // 0
		{ 0, 1, 1, 0, 0, 0, 0 },  // 1
		{ 1, 1, 0, 1, 1, 0, 1 },  // 2
		{ 1, 1, 1, 1, 0, 0, 1 },  // 3
		{ 0, 1, 1, 0, 0, 1, 1 },  // 4
		{ 1, 0, 1, 1, 0, 1, 1 },  // 5
		{ 0, 0, 1, 1, 1, 1, 1 },  // 6
		{ 1, 1, 1, 0, 0, 0, 0 },  // 7
		{ 1, 1, 1, 1, 1, 1, 1 },  // 8
		{ 1, 1, 1, 0, 0, 1, 1 }   // 9
};

const int Target[PatternCount][OutputNodes] = {
		{ 0, 0, 0, 0 },
		{ 0, 0, 0, 1 },
		{ 0, 0, 1, 0 },
		{ 0, 0, 1, 1 },
		{ 0, 1, 0, 0 },
		{ 0, 1, 0, 1 },
		{ 0, 1, 1, 0 },
		{ 0, 1, 1, 1 },
		{ 1, 0, 0, 0 },
		{ 1, 0, 0, 1 }
};

/******************************************************************
 * End Network Configuration
 ******************************************************************/

int i, j, p, q, r;
int ReportEvery1000;
int RandomizedIndex[PatternCount];
long TrainingCycle;
float Rando;
float Error;
float Accum;

float Hidden[HiddenNodes];
float Output[OutputNodes];
float HiddenWeights[InputNodes + 1][HiddenNodes];
float OutputWeights[HiddenNodes + 1][OutputNodes];
float HiddenDelta[HiddenNodes];
float OutputDelta[OutputNodes];
float ChangeHiddenWeights[InputNodes + 1][HiddenNodes];
float ChangeOutputWeights[HiddenNodes + 1][OutputNodes];

void toTerminal();
void calculateOutput(int p);
void saveWeightsToFile(string filename);
void readWeightsFromFile(string filename);
void saveCommaSeparated(ofstream &outFile, float data[], int n);
void trainFromFile(string filename);
void initHiddenAndChangeWeights();
void initOutputAndChangeWeights();
void train();


#include "NeuralNetwork.h"

void nnTest(){
	NeuralNetwork nn;
	//nn.trainFromFile("RGB.train");
	nn.readWeightsFromFile("7seg.mc");

	int data[] = {0,1,1,0,0,0,0}; //9  1001
	cout << "Prediction: " << nn.classify(data);


	exit(1);
}

int main() {

	nnTest();

	//trainFromFile("RGB.train");

	//readWeightsFromFile("7seg.mc");
	//calculateOutput(6);

	srand(time(NULL));
	ReportEvery1000 = 1;
	for (p = 0; p < PatternCount; p++) {
		RandomizedIndex[p] = p;
	}

	//while (true) {

	initHiddenAndChangeWeights();
	initOutputAndChangeWeights();
	train();

	toTerminal();

	cout << endl;
	cout << endl;
	cout << "TrainingCycle: ";
	cout << TrainingCycle;
	cout << "  Error = ";
	cout << Error << endl;

	toTerminal();

	cout << endl;
	cout << endl;
	cout << "Training Set Solved! " << endl;
	cout << "--------" << endl;
	cout << endl;
	cout << endl;
	ReportEvery1000 = 1;

	string filename = "7seg.mc";
	saveWeightsToFile(filename);

	cout << "Trying 1... ";
	calculateOutput(1);
	cout << endl;
	cout << "Trying 5... ";
	calculateOutput(5);
	cout << endl;
	cout << "Trying 3... ";
	calculateOutput(3);
	cout << endl;
	cout << "Trying 7... ";
	calculateOutput(7);
	cout << endl;

}

void trainFromFile(string filename) {

	ifstream inFile(filename);
	if (!inFile.is_open()) {
		cout << "Can't open input file" << endl;
		exit(1);
	}

	inFile >> numInputNodes;
	inFile >> numTrainingSamples;
	inFile >> numOutputNodes;
	inFile >> numClasses;
	//next call is getline and theres a newline sitting here
	inFile.ignore(10, '\n');

	cout << numInputNodes << endl;
	cout << numTrainingSamples << endl;
	cout << numOutputNodes << endl;
	cout << numClasses << endl;

	//load the class names
	classNames = new string[numClasses];
	for (int i = 0; i < numClasses; i++) {
		//read up to the comma if it's there
		if (i < numClasses - 1) {
			getline(inFile, classNames[i], ',');
		} else {
			getline(inFile, classNames[i]);
		}
		cout << classNames[i] << " ";
	}
	cout << endl;

	//load the inputNodes
	//Input[PatternCount][InputNodes]
	trainingData = (int**) malloc(sizeof(int) * numTrainingSamples);
	trainingClasses = (int**) malloc(sizeof(int) * numTrainingSamples);

	for (int r = 0; r < numTrainingSamples; ++r) {
			//read in the input nodes
		trainingData[r] = (int*) calloc(numInputNodes, sizeof(int));
		for (int c = 0; c < numInputNodes; ++c) {
			inFile >> trainingData[r][c];
			cout << trainingData[r][c] << " ";
		}
		//read in the classification nodes
		trainingClasses[r] = (int*) calloc(numOutputNodes, sizeof(int));
		for (int c = 0; c < numOutputNodes; ++c) {
			inFile >> trainingClasses[r][c];
			cout << trainingClasses[r][c] << " ";
		}
		cout << endl;
	}
	cout << "DONE" << endl;
	inFile.close();



}

void saveWeightsToFile(string filename) {
	//HiddenWeights 	 size: HiddenNodes
	//outputWeights  	 size: OutputNodes
	//Hidden
	//HiddenNodes:
	//OutputNodes:
	//Hidden Values comma separated
	//OutputValues comma separated
	//HiddenWeights values comma separated
	//OutputWeights values comma separated
	ofstream outFile(filename);
	if (!outFile.is_open()) {
		cout << "Can't open output file" << endl;
		exit(1);
	}
	outFile << HiddenNodes << endl;
	outFile << OutputNodes << endl;
	//Print the Hidden Weights
	//0 1 2
	//3 4 5
	//6 7 8
	for (int r = 0; r < HiddenNodes; ++r) {
		for (int c = 0; c < HiddenNodes; ++c) {
			outFile << HiddenWeights[r][c];
			//print comma if not the last value
			//(r*HiddenNodes) + c : this will turn the 2D indexing
			//into 1D
			if ((r * HiddenNodes) + c != (HiddenNodes * HiddenNodes) - 1) {
				outFile << " ";
			}
		}
	}
	outFile << endl;
	//Print the output weights
	//float OutputWeights[HiddenNodes + 1][OutputNodes];
	for (int r = 0; r < HiddenNodes + 1; ++r) {
		for (int c = 0; c < OutputNodes; ++c) {
			outFile << OutputWeights[r][c];
			//print comma if not the last value
			//(r*HiddenNodes) + c : this will turn the 2D indexing
			//into 1D
			if ((r * OutputNodes) + c
					!= ((HiddenNodes + 1) * OutputNodes) - 1) {
				outFile << " ";
			}
		}
	}
	outFile << endl;
	outFile.close();

}

void readWeightsFromFile(string filename) {

	ifstream inFile(filename);
	if (!inFile.is_open()) {
		cout << "Can't open input file" << endl;
		exit(1);
	}

	inFile >> numHiddenNodes;
	inFile >> numOutputNodes;
	//cout << fileHiddenNodes << endl;
	//cout << fileOutputNodes << endl;
	//read in hidden weights
	for (int r = 0; r < numHiddenNodes; ++r) {
		for (int c = 0; c < numHiddenNodes; ++c) {
			inFile >> HiddenWeights[r][c];
			//cout << HiddenWeights[r][c] << " ";
		}
	}
	cout << endl;
	//read in output weights
	for (int r = 0; r < numHiddenNodes + 1; ++r) {
		for (int c = 0; c < numOutputNodes; ++c) {
			inFile >> OutputWeights[r][c];
			//cout << OutputWeights[r][c] << " ";
		}
	}

	cout << endl << "Loaded weights from " << filename << endl;

	inFile.close();
}

void saveCommaSeparated(ofstream &outFile, float data[], int n) {
	for (int i = 0; i < n; ++i) {
		outFile << data[i];
		if (i != (n - 1)) {
			outFile << " ";
		}
	}
	outFile << endl;
}

//pass in a pattern to calculate the output for
void calculateOutput(int p) {
	/******************************************************************
	 * Compute hidden layer activations
	 ******************************************************************/

	for (i = 0; i < HiddenNodes; i++) {
		Accum = HiddenWeights[InputNodes][i];
		for (j = 0; j < InputNodes; j++) {
			Accum += Input[p][j] * HiddenWeights[j][i];
		}
		Hidden[i] = 1.0 / (1.0 + exp(-Accum));
	}

	/******************************************************************
	 * Compute output layer activations and calculate errors
	 ******************************************************************/

	for (i = 0; i < OutputNodes; i++) {
		Accum = OutputWeights[HiddenNodes][i];
		for (j = 0; j < HiddenNodes; j++) {
			Accum += Hidden[j] * OutputWeights[j][i];
		}
		Output[i] = 1.0 / (1.0 + exp(-Accum));
	}
	cout << "  Output ";
	for (i = 0; i < OutputNodes; i++) {
		cout << int(Output[i] + 0.5) << " ";
	}
}

void toTerminal() {

	for (p = 0; p < PatternCount; p++) {
		cout << endl;
		cout << "  Training Pattern: ";
		cout << p ;
		cout << "  Input ";
		for (i = 0; i < InputNodes; i++) {
			cout << Input[p][i] << " ";
		}
		cout << "  Target ";
		for (i = 0; i < OutputNodes; i++) {
			cout << Target[p][i] << " ";
		}

		//This runs the input nodes through the current network?
		//Need

		/******************************************************************
		 * Compute hidden layer activations
		 ******************************************************************/

		for (i = 0; i < HiddenNodes; i++) {
			Accum = HiddenWeights[InputNodes][i];
			for (j = 0; j < InputNodes; j++) {
				Accum += Input[p][j] * HiddenWeights[j][i];
			}
			Hidden[i] = 1.0 / (1.0 + exp(-Accum));
		}

		/******************************************************************
		 * Compute output layer activations and calculate errors
		 ******************************************************************/

		for (i = 0; i < OutputNodes; i++) {
			Accum = OutputWeights[HiddenNodes][i];
			for (j = 0; j < HiddenNodes; j++) {
				Accum += Hidden[j] * OutputWeights[j][i];
			}
			Output[i] = 1.0 / (1.0 + exp(-Accum));
		}
		cout << "  Output ";
		for (i = 0; i < OutputNodes; i++) {
			cout << Output[i] << " ";
		}
	}

}

/******************************************************************
 * Initialize HiddenWeights and ChangeHiddenWeights
 ******************************************************************/
void initHiddenAndChangeWeights() {
	for (i = 0; i < HiddenNodes; i++) {
		for (j = 0; j <= InputNodes; j++) {
			ChangeHiddenWeights[j][i] = 0.0;
			Rando = float(rand() % 100) / 100;
			HiddenWeights[j][i] = 2.0 * (Rando - 0.5) * InitialWeightMax;
		}
	}
}

/******************************************************************
 * Initialize OutputWeights and ChangeOutputWeights
 ******************************************************************/
void initOutputAndChangeWeights() {
	for (i = 0; i < OutputNodes; i++) {
		for (j = 0; j <= HiddenNodes; j++) {
			ChangeOutputWeights[j][i] = 0.0;
			Rando = float(rand() % 100) / 100;
			OutputWeights[j][i] = 2.0 * (Rando - 0.5) * InitialWeightMax;
		}
	}
}

void train() {
	/******************************************************************
	 * Begin training
	 ******************************************************************/

	for (TrainingCycle = 1; TrainingCycle < 2147483647; TrainingCycle++) {

		/******************************************************************
		 * Randomize order of training patterns
		 ******************************************************************/

		for (p = 0; p < PatternCount; p++) {
			q = rand() % PatternCount;
			r = RandomizedIndex[p];
			RandomizedIndex[p] = RandomizedIndex[q];
			RandomizedIndex[q] = r;
		}
		Error = 0.0;
		/******************************************************************
		 * Cycle through each training pattern in the randomized order
		 ******************************************************************/
		for (q = 0; q < PatternCount; q++) {
			p = RandomizedIndex[q];

			/******************************************************************
			 * Compute hidden layer activations
			 ******************************************************************/

			for (i = 0; i < HiddenNodes; i++) {
				Accum = HiddenWeights[InputNodes][i];
				for (j = 0; j < InputNodes; j++) {
					Accum += Input[p][j] * HiddenWeights[j][i];
				}
				Hidden[i] = 1.0 / (1.0 + exp(-Accum));
			}

			/******************************************************************
			 * Compute output layer activations and calculate errors
			 ******************************************************************/

			for (i = 0; i < OutputNodes; i++) {
				Accum = OutputWeights[HiddenNodes][i];
				for (j = 0; j < HiddenNodes; j++) {
					Accum += Hidden[j] * OutputWeights[j][i];
				}
				Output[i] = 1.0 / (1.0 + exp(-Accum));
				OutputDelta[i] = (Target[p][i] - Output[i]) * Output[i]
						* (1.0 - Output[i]);
				Error += 0.5 * (Target[p][i] - Output[i])
						* (Target[p][i] - Output[i]);
			}

			/******************************************************************
			 * Backpropagate errors to hidden layer
			 ******************************************************************/

			for (i = 0; i < HiddenNodes; i++) {
				Accum = 0.0;
				for (j = 0; j < OutputNodes; j++) {
					Accum += OutputWeights[i][j] * OutputDelta[j];
				}
				HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]);
			}

			/******************************************************************
			 * Update Inner-->Hidden Weights
			 ******************************************************************/

			for (i = 0; i < HiddenNodes; i++) {
				ChangeHiddenWeights[InputNodes][i] = LearningRate
						* HiddenDelta[i]
						+ Momentum * ChangeHiddenWeights[InputNodes][i];
				HiddenWeights[InputNodes][i] +=
						ChangeHiddenWeights[InputNodes][i];
				for (j = 0; j < InputNodes; j++) {
					ChangeHiddenWeights[j][i] = LearningRate * Input[p][j]
							* HiddenDelta[i]
							+ Momentum * ChangeHiddenWeights[j][i];
					HiddenWeights[j][i] += ChangeHiddenWeights[j][i];
				}
			}

			/******************************************************************
			 * Update Hidden-->Output Weights
			 ******************************************************************/

			for (i = 0; i < OutputNodes; i++) {
				ChangeOutputWeights[HiddenNodes][i] = LearningRate
						* OutputDelta[i]
						+ Momentum * ChangeOutputWeights[HiddenNodes][i];
				OutputWeights[HiddenNodes][i] +=
						ChangeOutputWeights[HiddenNodes][i];
				for (j = 0; j < HiddenNodes; j++) {
					ChangeOutputWeights[j][i] = LearningRate * Hidden[j]
							* OutputDelta[i]
							+ Momentum * ChangeOutputWeights[j][i];
					OutputWeights[j][i] += ChangeOutputWeights[j][i];
				}
			}
		} //end For Pattern Count

		/******************************************************************
		 * Every 1000 cycles send data to terminal for display
		 ******************************************************************/
		ReportEvery1000 = ReportEvery1000 - 1;
		if (ReportEvery1000 == 0) {
			cout << endl;
			cout << endl;
			cout << "TrainingCycle: ";
			cout << TrainingCycle;
			cout << "  Error = ";
			cout << Error << endl;

			toTerminal();

			if (TrainingCycle == 1) {
				ReportEvery1000 = 999;
			} else {
				ReportEvery1000 = 1000;
			}
		} //end if report

		/******************************************************************
		 * If error rate is less than pre-determined threshold then end
		 ******************************************************************/

		if (Error < Success)
			break;
	} //end training
}


