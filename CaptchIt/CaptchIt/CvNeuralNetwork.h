#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/ml/ml.hpp"

#include "CvTools.h"

#define TRAINING_SAMPLES 3050       //Number of samples in training dataset
#define TEST_SAMPLES 1170       //Number of samples in test dataset

using namespace std;
using namespace cv;

class CvNeuralNetwork
{
public:
	CvNeuralNetwork(string datasetPath, string trainingPath, string testPath, string nnParametersPath);
	~CvNeuralNetwork();

	void setDebugMode(bool debugMode);
	void testParameters();
	int predictClass(string filename);

	// Number of pixels per sample (256 = 16x16)
	static const int ATTRIBUTES = 256;

	// Number of classes
	static const int CLASSES = 10;

protected:
	void computeParameters();
	void loadParameters();

private:
	string datasetPath;
	string trainingPath;
	string testPath;
	string nnParametersPath;
	
	bool debugMode;

	CvANN_MLP neuralNetwork;
};

