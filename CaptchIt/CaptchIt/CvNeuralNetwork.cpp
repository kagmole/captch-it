#include "stdafx.h"
#include <stdio.h>
#include "CvNeuralNetwork.h"


CvNeuralNetwork::CvNeuralNetwork(string datasetPath, string trainingPath, string testPath, string nnParametersPath)
{
	// Disable debug mode
	this->debugMode = false;

	// Dataset folder path
	this->datasetPath = datasetPath;

	// Training set file
	this->trainingPath = trainingPath;

	// Test set file
	this->testPath = testPath;

	// Neural Network parameters file
	this->nnParametersPath = nnParametersPath;

	if (CvTools::fileExists(this->nnParametersPath))
	{
		this->loadParameters();
	}
	else
	{
		this->computeParameters();
	}
}

CvNeuralNetwork::~CvNeuralNetwork()
{
}

void CvNeuralNetwork::setDebugMode(bool debugMode)
{
	this->debugMode = debugMode;
}

void CvNeuralNetwork::loadParameters()
{
	if (debugMode) cout << "CvNeuralNetwork::loadParametersFromFile() START" << endl;

	CvFileStorage* storage = cvOpenFileStorage((char*)this->nnParametersPath.c_str(), 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "DigitOCR");
	this->neuralNetwork.read(storage, n);
	cvReleaseFileStorage(&storage);

	if (debugMode) cout << "CvNeuralNetwork::loadParametersFromFile() END" << endl;
}

void CvNeuralNetwork::computeParameters()
{
	if (debugMode) cout << "CvNeuralNetwork::computeParameters() START" << endl;

	Mat training_set(TRAINING_SAMPLES, CvNeuralNetwork::ATTRIBUTES, CV_32F);
	Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);

	CvTools::readDataset((char*)trainingPath.c_str(), training_set, training_set_classifications, TRAINING_SAMPLES);

	// The neural network has 3 layers :
	// - 256 inputs nodes, one for each pixel of the normalized image
	// - 16 hidden nodes
	// - 10 output node, one for each class
	Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = CvNeuralNetwork::ATTRIBUTES; // Input layer
	layers.at<int>(1, 0) = 16; // Hidden layer
	layers.at<int>(2, 0) = CLASSES; // Output layer

	// Create the neural network (http://docs.opencv.org/modules/ml/doc/neural_networks.html)
	if (debugMode) cout << "Creation of the Neural Network ..." << endl;
	CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

	// Neural network parameters
	CvANN_MLP_TrainParams params(

		// Maximum 1000 iterations for training or a small change
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001),

		// Backpropogation for training
		CvANN_MLP_TrainParams::BACKPROP,

		// Coefficents for backpropogation training (http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams)
		0.1,
		0.1
	);

	// Train the neural network (using training data)
	if (debugMode) cout << "Training the Neural Newtork ..." << endl;
	int iterations = nnetwork.train(training_set, training_set_classifications, cv::Mat(), cv::Mat(), params);
	if (debugMode) cout << "Training finished. " << iterations << " iterations" << endl;

	// Save the model generated into an xml file
	if (debugMode) cout << "Saving parameters of the Neural Network ..." << endl;
	CvFileStorage* storage = cvOpenFileStorage((char*)this->nnParametersPath.c_str(), 0, CV_STORAGE_WRITE);
	nnetwork.write(storage, "DigitOCR");
	cvReleaseFileStorage(&storage);

	if (debugMode) cout << "CvNeuralNetwork::computeParameters() END" << endl;
}

void CvNeuralNetwork::testParameters()
{
	if (debugMode) cout << "CvNeuralNetwork::testParameters() START" << endl;

	Mat test_set(TEST_SAMPLES, CvNeuralNetwork::ATTRIBUTES, CV_32F);
	Mat test_set_classifications(TEST_SAMPLES, CLASSES, CV_32F);

	Mat classificationResult(1, CLASSES, CV_32F);

	CvTools::readDataset((char*)testPath.c_str(), test_set, test_set_classifications, TEST_SAMPLES);

	cv::Mat test_sample;
	int correct_class = 0;
	int wrong_class = 0;

	// Classification matrix gives the count of classes to which the samples were classified.
	int classification_matrix[CLASSES][CLASSES] = { {} };

	// Foreach sample in test dataset
	for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {

		// Extract the sample
		test_sample = test_set.row(tsample);

		// Try to predict its class
		this->neuralNetwork.predict(test_sample, classificationResult);
		// The classification result matrix holds weightage of each class.
		// We take the class with the highest weightage as the resultant class

		// Find the class with maximum weightage
		int maxIndex = CvTools::searchMaxWeight(classificationResult);

		if (debugMode) cout << "Testing Sample " << tsample << " -> Class " << maxIndex << endl;

		// Check if the predicted class is correct
		if (test_set_classifications.at<float>(tsample, maxIndex) != 1.0f)
		{
			wrong_class++;

			// Find the actual class
			for (int class_index = 0; class_index < CvNeuralNetwork::CLASSES; class_index++)
			{
				if (test_set_classifications.at<float>(tsample, class_index) == 1.0f)
				{
					classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
					break;
				}
			}

		}
		else 
		{
			correct_class++;
			classification_matrix[maxIndex][maxIndex]++;
		}
	}

	printf("\nResults on the testing dataset\n"
		"\tCorrect classification: %d (%g%%)\n"
		"\tWrong classifications: %d (%g%%)\n",
		correct_class, (double)correct_class * 100 / TEST_SAMPLES,
		wrong_class, (double)wrong_class * 100 / TEST_SAMPLES);

	cout << "   ";
	for (int i = 0; i < CLASSES; i++)
	{
		cout << i << "\t";
	}
	cout << "\n";
	for (int row = 0; row<CLASSES; row++)
	{
		cout << row << "  ";
		for (int col = 0; col<CLASSES; col++)
		{
			cout << classification_matrix[row][col] << "\t";
		}
		cout << "\n";
	}

	if (debugMode) cout << "CvNeuralNetwork::testParameters() END" << endl;
}

int CvNeuralNetwork::predictClass(string filename)
{
	int imageDataPixelValue[256];
	CvTools::imageToPixelValue(filename, imageDataPixelValue);

	Mat data(1, CvNeuralNetwork::ATTRIBUTES, CV_32F);

	CvTools::loadDataToMat(imageDataPixelValue, data);

	int maxIndex = 0;
	Mat classOut(1, CvNeuralNetwork::CLASSES, CV_32F);

	// Prediction
	this->neuralNetwork.predict(data, classOut);

	return CvTools::searchMaxWeight(classOut);
}
