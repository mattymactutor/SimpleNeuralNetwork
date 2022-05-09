#include "NeuralNetwork.h"
#include <iostream>
using namespace std;

NeuralNetwork::NeuralNetwork() {

}
void NeuralNetwork::trainFromFile(string filename) {

	ifstream inFile(filename);
	if (!inFile.is_open()) {
		cout << "Can't open input file" << endl;
		exit(1);
	}

	inFile >> numInputNodes;
	inFile >> numOutputNodes;
	inFile >> numHiddenNodes;
	inFile >> numTrainingSamples;
	inFile >> numClasses;
	//next call is getline and theres a newline sitting here
	inFile.ignore(10, '\n');


	//load the class names
	classNames = new string[numClasses];
	for (int i = 0; i < numClasses; i++) {
		//read up to the comma if it's there
		if (i < numClasses - 1) {
			getline(inFile, classNames[i], ',');
		} else {
			getline(inFile, classNames[i]);
		}
	}


	//load the inputNodes
	//Input[PatternCount][InputNodes]
	trainingData = (int**) malloc(sizeof(int) * numTrainingSamples);
	trainingClasses = (int**) malloc(sizeof(int) * numTrainingSamples);

	for (int r = 0; r < numTrainingSamples; ++r) {
		//read in the input nodes
		trainingData[r] = (int*) calloc(numInputNodes, sizeof(int));
		for (int c = 0; c < numInputNodes; ++c) {
			inFile >> trainingData[r][c];
		}
		//read in the classification nodes
		trainingClasses[r] = (int*) calloc(numOutputNodes, sizeof(int));
		for (int c = 0; c < numOutputNodes; ++c) {
			inFile >> trainingClasses[r][c];
		}
	}
	printCurrentModel();
	inFile.close();

	RandomizedIndex = (int *) calloc(numTrainingSamples,sizeof(int));
	for (int p = 0; p < numTrainingSamples; p++) {
		RandomizedIndex[p] = p;
	}

	//allocate space for all other arrays
	Hidden = (float *) calloc(numHiddenNodes,sizeof(float)); //[HiddenNodes];
	Output = (float *) calloc(numOutputNodes, sizeof(float));  //[OutputNodes];

	allocate2D(&HiddenWeights, numInputNodes + 1,numHiddenNodes); //[InputNodes + 1][HiddenNodes];
	allocate2D(&OutputWeights, numHiddenNodes + 1,numOutputNodes); //[HiddenNodes + 1][OutputNodes];
	HiddenDelta = (float *) calloc(numHiddenNodes,sizeof(float)); //[HiddenNodes];
	OutputDelta = (float *) calloc(numOutputNodes,sizeof(float));; //[OutputNodes];
	allocate2D(&ChangeHiddenWeights, numInputNodes + 1,numHiddenNodes); //[InputNodes + 1][HiddenNodes];
	allocate2D(&ChangeOutputWeights, numHiddenNodes + 1,numOutputNodes); //[HiddenNodes + 1][OutputNodes];

	initHiddenAndChangeWeights();
	initOutputAndChangeWeights();
	train();
	//change the format for the output
	string outFilename = filename.substr(0,filename.find(".")) + ".mc";
	saveWeightsToFile(outFilename);

}

void NeuralNetwork::saveWeightsToFile(string filename){

	ofstream outFile(filename);
	if (!outFile.is_open()) {
		cout << "Can't open output file" << endl;
		exit(1);
	}
	outFile << numInputNodes << endl;
	outFile << numOutputNodes << endl;
	outFile << numHiddenNodes << endl;
	outFile << numClasses << endl;
	for (int i = 0; i < numClasses; ++i) {
		outFile << classNames[i];
		if (i != numClasses -1 ) outFile << ",";
	}
	outFile << endl;

	//Print the Hidden Weights
	for (int r = 0; r < numInputNodes+1; ++r) {
		for (int c = 0; c < numHiddenNodes; ++c) {
			outFile << HiddenWeights[r][c];
			//print comma if not the last value
			//(r*HiddenNodes) + c : this will turn the 2D indexing
			//into 1D
			if ((r * numHiddenNodes) + c != (numHiddenNodes * numHiddenNodes) - 1) {
				outFile << " ";
			}
		}
	}
	outFile << endl;
	//Print the output weights
	//float OutputWeights[HiddenNodes + 1][OutputNodes];
	for (int r = 0; r < numHiddenNodes + 1; ++r) {
		for (int c = 0; c < numOutputNodes; ++c) {
			outFile << OutputWeights[r][c];
			//print comma if not the last value
			//(r*HiddenNodes) + c : this will turn the 2D indexing
			//into 1D
			if ((r * numOutputNodes) + c
					!= ((numHiddenNodes + 1) * numOutputNodes) - 1) {
				outFile << " ";
			}
		}
	}
	outFile << endl;
	outFile.close();
	cout << "Saved weights to " << filename << endl;

}
void NeuralNetwork::allocate2D( float *** ptr, int rows, int cols){
	float ** out =(float **) calloc(rows,sizeof(float *));
	for (int r = 0; r < rows; r++) {
		out[r] = (float*) calloc(cols,sizeof(float));
	}
	*ptr = out;
}

string NeuralNetwork::classify(int *data) {
/******************************************************************
	 * Compute hidden layer activations
	 ******************************************************************/

	for (int i = 0; i < numHiddenNodes; i++) {
		Accum = HiddenWeights[numInputNodes][i];
		for (int j = 0; j < numInputNodes; j++) {
			Accum += data[j] * HiddenWeights[j][i];
		}
		Hidden[i] = 1.0 / (1.0 + exp(-Accum));
	}

	/******************************************************************
	 * Compute output layer activations and calculate errors
	 ******************************************************************/

	for (int i = 0; i < numOutputNodes; i++) {
		Accum = OutputWeights[numHiddenNodes][i];
		for (int j = 0; j < numHiddenNodes; j++) {
			Accum += Hidden[j] * OutputWeights[j][i];
		}
		Output[i] = 1.0 / (1.0 + exp(-Accum));
	}
	//cout << "  Output ";
	for (int i = 0; i < numOutputNodes; i++) {
		//cout << int(Output[i] + 0.5) << " ";
	}
	//find the output in the list of classes
	//the output should be a number in binary so convert it
	int sum = 0;
	int exp = 0;
	for(int i = numOutputNodes - 1; i >= 0 ; i--){
		//round to 0 or 1
		int bit = Output[i] + 0.5;
		sum += bit * pow(2,exp++);
	}
	//cout << "Prediction: " << sum << endl;
	return classNames[sum];
}

void NeuralNetwork::printCurrentModel(){

	cout << "Num Input Nodes: " << numInputNodes << endl;
	cout << "Num Output Nodes: " << numOutputNodes << endl;
	cout << "Num Hidden Nodes:" << numHiddenNodes << endl;
	cout << "Num Training Samples: " << numTrainingSamples << endl;
	cout << "Num Classes: " << numClasses << endl;
	cout << "Classes: ";
	for (int i = 0; i < numClasses; ++i) {
		cout << classNames[i] << ", ";
	}
	cout << endl;
	cout << "-Training Data        | Class ----" << endl;
	for (int i = 0; i < numTrainingSamples; ++i) {
		for (int j = 0; j < numInputNodes; j++){
			cout << trainingData[i][j] << " ";
		}
		cout << "\t";
		//also print the classification
		for (int j = 0; j < numOutputNodes; ++j) {
			cout << trainingClasses[i][j] << " ";
		}
		cout << endl;
	}


}

void NeuralNetwork::initHiddenAndChangeWeights() {
	cout << "Initializing hidden weights..." << endl;
	//float ChangeHiddenWeights[InputNodes + 1][HiddenNodes];
	for (int i = 0; i < numHiddenNodes; i++) {
		for (int j = 0; j <= numInputNodes; j++) {
			ChangeHiddenWeights[j][i] = 0.0;
			Rando = float(rand() % 100) / 100;
			HiddenWeights[j][i] = 2.0 * (Rando - 0.5) * InitialWeightMax;
		}
	}
}
void NeuralNetwork::initOutputAndChangeWeights() {
	cout << "Initializing output weights..." << endl;
	for (int i = 0; i < numOutputNodes; i++) {
		for (int j = 0; j <= numHiddenNodes; j++) {
			ChangeOutputWeights[j][i] = 0.0;
			Rando = float(rand() % 100) / 100;
			OutputWeights[j][i] = 2.0 * (Rando - 0.5) * InitialWeightMax;
		}
	}
}
void NeuralNetwork::train() {
	cout << "Started training." << endl;
/******************************************************************
	 * Begin training
	 ******************************************************************/
	int q,r,i,j,p;
	for (TrainingCycle = 1; TrainingCycle < 2147483647; TrainingCycle++) {

		/******************************************************************
		 * Randomize order of training patterns
		 ******************************************************************/

		for (p = 0; p < numTrainingSamples; p++) {
			q = rand() % numTrainingSamples;
			r = RandomizedIndex[p];
			RandomizedIndex[p] = RandomizedIndex[q];
			RandomizedIndex[q] = r;
		}
		Error = 0.0;
		/******************************************************************
		 * Cycle through each training pattern in the randomized order
		 ******************************************************************/
		for (q = 0; q < numTrainingSamples; q++) {
			p = RandomizedIndex[q];

			/******************************************************************
			 * Compute hidden layer activations
			 ******************************************************************/

			for (i = 0; i < numHiddenNodes; i++) {
				Accum = HiddenWeights[numInputNodes][i];
				for (j = 0; j < numInputNodes; j++) {
					Accum += trainingData[p][j] * HiddenWeights[j][i];
				}
				Hidden[i] = 1.0 / (1.0 + exp(-Accum));
			}

			/******************************************************************
			 * Compute output layer activations and calculate errors
			 ******************************************************************/

			for (i = 0; i < numOutputNodes; i++) {
				Accum = OutputWeights[numHiddenNodes][i];
				for (j = 0; j < numHiddenNodes; j++) {
					Accum += Hidden[j] * OutputWeights[j][i];
				}
				Output[i] = 1.0 / (1.0 + exp(-Accum));
				OutputDelta[i] = (trainingClasses[p][i] - Output[i]) * Output[i]
						* (1.0 - Output[i]);
				Error += 0.5 * (trainingClasses[p][i] - Output[i])
						* (trainingClasses[p][i] - Output[i]);
			}

			/******************************************************************
			 * Backpropagate errors to hidden layer
			 ******************************************************************/

			for (i = 0; i < numHiddenNodes; i++) {
				Accum = 0.0;
				for (j = 0; j < numOutputNodes; j++) {
					Accum += OutputWeights[i][j] * OutputDelta[j];
				}
				HiddenDelta[i] = Accum * Hidden[i] * (1.0 - Hidden[i]);
			}

			/******************************************************************
			 * Update Inner-->Hidden Weights
			 ******************************************************************/

			for (i = 0; i < numHiddenNodes; i++) {
				ChangeHiddenWeights[numInputNodes][i] = LearningRate
						* HiddenDelta[i]
						+ Momentum * ChangeHiddenWeights[numInputNodes][i];
				HiddenWeights[numInputNodes][i] +=
						ChangeHiddenWeights[numInputNodes][i];
				for (j = 0; j < numInputNodes; j++) {
					ChangeHiddenWeights[j][i] = LearningRate * trainingData[p][j]
							* HiddenDelta[i]
							+ Momentum * ChangeHiddenWeights[j][i];
					HiddenWeights[j][i] += ChangeHiddenWeights[j][i];
				}
			}

			/******************************************************************
			 * Update Hidden-->Output Weights
			 ******************************************************************/

			for (i = 0; i < numOutputNodes; i++) {
				ChangeOutputWeights[numHiddenNodes][i] = LearningRate
						* OutputDelta[i]
						+ Momentum * ChangeOutputWeights[numHiddenNodes][i];
				OutputWeights[numHiddenNodes][i] +=
						ChangeOutputWeights[numHiddenNodes][i];
				for (j = 0; j < numHiddenNodes; j++) {
					ChangeOutputWeights[j][i] = LearningRate * Hidden[j]
							* OutputDelta[i]
							+ Momentum * ChangeOutputWeights[j][i];
					OutputWeights[j][i] += ChangeOutputWeights[j][i];
				}
			}
		} //end For Pattern Count

		if(TrainingCycle % 1000 == 0){
			cout << endl << endl;
			cout << "--TrainingCycle: " << TrainingCycle <<" Error: " << Error << endl;
			printTrainData();
		}

		/******************************************************************
		 * If error rate is less than pre-determined threshold then end
		 ******************************************************************/

		if (Error < Success){
			cout << endl << endl;
			cout << "**COMPLETE**   TrainingCycle: " << TrainingCycle <<" Error: " << Error << endl;
			printTrainData();
			break;
		}
	} //end training

}

void NeuralNetwork::printTrainData(){
	for (int p = 0; p < numTrainingSamples; p++) {

		cout << "Training Pattern: ";
		cout << p ;
		cout << "  Input ";
		for (int i = 0; i < numInputNodes; i++) {
			cout << trainingData[p][i] << " ";
		}
		cout << "  Target ";
		for (int i = 0; i < numOutputNodes; i++) {
			cout << trainingClasses[p][i] << " ";
		}

		//This runs the input nodes through the current network?
		//Need

		/******************************************************************
		 * Compute hidden layer activations
		 ******************************************************************/

		for (int i = 0; i < numHiddenNodes; i++) {
			Accum = HiddenWeights[numInputNodes][i];
			for (int j = 0; j < numInputNodes; j++) {
				Accum += trainingData[p][j] * HiddenWeights[j][i];
			}
			Hidden[i] = 1.0 / (1.0 + exp(-Accum));
		}

		/******************************************************************
		 * Compute output layer activations and calculate errors
		 ******************************************************************/

		for (int i = 0; i < numOutputNodes; i++) {
			Accum = OutputWeights[numHiddenNodes][i];
			for (int j = 0; j < numHiddenNodes; j++) {
				Accum += Hidden[j] * OutputWeights[j][i];
			}
			Output[i] = 1.0 / (1.0 + exp(-Accum));
		}
		cout << "  Output ";
		for (int i = 0; i < numOutputNodes; i++) {
			cout << Output[i] << " ";
		}
		cout << endl;
	}
}

void freeArray2D(float ** ptr, int rows, int cols){

	for (int r = 0; r < rows; r++) {
		delete[] ptr[r];
	}
	delete ptr;
}

void NeuralNetwork::readWeightsFromFile(string filename){
	ifstream inFile(filename);
	if (!inFile.is_open()) {
		cout << "Can't open input file" << endl;
		exit(1);
	}

	if (HiddenWeights != nullptr) {
		freeArray2D(HiddenWeights,numInputNodes + 1,numHiddenNodes);
	}
	if (OutputWeights != nullptr){
		freeArray2D(OutputWeights, numHiddenNodes + 1,numOutputNodes);
	}
	inFile >> numInputNodes;
	inFile >> numOutputNodes;
	inFile >> numHiddenNodes;
	inFile >> numClasses;
	//next call is getline and theres a newline sitting here
	inFile.ignore(10, '\n');

	if (classNames != nullptr){
		delete[] classNames;
	}
	//load the class names
	classNames = new string[numClasses];
	for (int i = 0; i < numClasses; i++) {
		//read up to the comma if it's there
		if (i < numClasses - 1) {
			getline(inFile, classNames[i], ',');
		} else {
			getline(inFile, classNames[i]);
		}
	}


	Hidden = (float *) calloc(numHiddenNodes,sizeof(float)); //[HiddenNodes];
	Output = (float *) calloc(numOutputNodes, sizeof(float));  //[OutputNodes];
	allocate2D(&HiddenWeights, numInputNodes + 1,numHiddenNodes); //[InputNodes + 1][HiddenNodes];
	allocate2D(&OutputWeights, numHiddenNodes + 1,numOutputNodes); //[HiddenNodes + 1][OutputNodes];



	//read in hidden weights
	for (int r = 0; r < numInputNodes+1; ++r) {
		for (int c = 0; c < numHiddenNodes; ++c) {
			inFile >> HiddenWeights[r][c];
		}
	}

	//read in output weights
	for (int r = 0; r < numHiddenNodes + 1; ++r) {
		for (int c = 0; c < numOutputNodes; ++c) {
			inFile >> OutputWeights[r][c];
		}
	}

	cout << endl << "Loaded weights from " << filename << endl;

	inFile.close();
}
