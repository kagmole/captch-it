#include "stdafx.h"
#include <stdlib.h>
#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <string.h>
#include <fstream>
#include <io.h>
//#include <sys/param.h>

#include "opencv2/ml/ml.hpp"          // opencv machine learning include file

#pragma warning (disable : 4996)
#include <stdio.h>

#define ATTRIBUTES 256  //Number of pixels per sample.16X16
#define CLASSES 10                  //Number of distinct labels.
#define TRAINING_SAMPLES 3050       //Number of samples in training dataset
#define TEST_SAMPLES 1170       //Number of samples in test dataset

using namespace cv;
using namespace std;

void scaleDownImage(cv::Mat &originalImg, cv::Mat &scaledDownImage)
{
	for (int x = 0; x<16; x++)
	{
		for (int y = 0; y<16; y++)
		{
			int yd = ceil((float)(y*originalImg.cols / 16));
			int xd = ceil((float)(x*originalImg.rows / 16));
			scaledDownImage.at<uchar>(x, y) = originalImg.at<uchar>(xd, yd);
		}
	}
}
/*
cv::Rect findLargestRect(cv::Mat &src)
{
	int thresh = 100;
	Mat threshold_output;
	vector<vector<Point>> contours;
	vector<Vec4i> hierarchy;

	// Detect edges using Threshold
	threshold(src, threshold_output, thresh, 255, THRESH_BINARY);

	// Find contours
	findContours(threshold_output, contours, hierarchy, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

	// Finds the contour with the largest area
	int area = 0;
	int idx;

	for (int i = 0; i<contours.size(); i++) {
		if (area < contours[i].size())
			idx = i;
	}

	// Calculates the bounding rect of the largest area contour
	cv::Rect rect = boundingRect(contours[idx]);

	return rect;
}

void cropAuto(cv::Mat &originalImage, cv::Mat &croppedImage)
{
	cout << "Test";
	cv::Rect rect = findLargestRect(originalImage);
	cout << "Test";
	cv::Mat crop(originalImage, rect);
	croppedImage = crop.clone();
}
*/
void cropImage(cv::Mat &originalImage, cv::Mat &croppedImage)
{
	int row = originalImage.rows;
	int col = originalImage.cols;
	int tlx, tly, bry, brx;//t=top r=right b=bottom l=left
	tlx = tly = bry = brx = 0;
	float suml = 0;
	float sumr = 0;
	int flag = 0;


	// Top Edge
	for (int x = 1; x<row; x++)
	{
		for (int y = 0; y<col; y++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{

				flag = 1;
				tly = x;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}

	}

	// Bottom Edge
	for (int x = row - 1; x>0; x--)
	{
		for (int y = 0; y<col; y++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{

				flag = 1;
				bry = x;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}

	}

	// Left
	for (int y = 0; y<col; y++)
	{
		for (int x = 0; x<row; x++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{

				flag = 1;
				tlx = y;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}

	// Right
	for (int y = col - 1; y>0; y--)
	{
		for (int x = 0; x<row; x++)
		{
			if (originalImage.at<uchar>(x, y) == 0)
			{

				flag = 1;
				brx = y;
				break;
			}

		}
		if (flag == 1)
		{
			flag = 0;
			break;
		}
	}

	int width = brx - tlx;
	int height = bry - tly;
	cv::Mat crop(originalImage, cv::Rect(tlx, tly, brx - tlx, bry - tly));
	croppedImage = crop.clone();

}

void pixelToValue(cv::Mat &img, int pixelArray[])
{
	int i = 0;
	for (int x = 0; x < 16; ++x)
	{
		for (int y = 0; y < 16; ++y)
		{
			// Image is thresholded :
			// Value = 1 if black pixel (255), 0 otherwise (white pixel)
			pixelArray[i] = (img.at<uchar>(x, y) == 255) ? 1 : 0;
			++i;
		}
	}
}

string intToString(int number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}

string pixelValueToString(int pixelArray[])
{
	string finalString;

	for (int i = 0; i < ATTRIBUTES; ++i)
	{
		finalString += intToString(pixelArray[i]);
	}

	return finalString;
}

cv::Mat parseImage(std::string imagePath)
{
	// Load image in a matrix
	cv::Mat img = cv::imread(imagePath, 0);
	cv::Mat output;

	// Remove noise
	cv::GaussianBlur(img, output, cv::Size(5, 5), 0);

	// Transform to binary image
	cv::threshold(output, output, 50, 255, 0);

	// Variable to store scaled image
	cv::Mat scaledDownImage(16, 16, CV_8U, cv::Scalar(0));

	// Crop image
	cropImage(output, output);

	// Scale down image to 16x16
	scaleDownImage(output, scaledDownImage);

	return scaledDownImage;
}

void writePixelValue(fstream &file, int* pixelValue, int character)
{
	for (int pixelIndex = 0; pixelIndex < 256; ++pixelIndex) 
	{
		file << pixelValue[pixelIndex] << ",";
	}

	// Write the label at the end of the data
	file << character << "\n";
}

void imageToPixelValue(std::string imagePath, int* pixelValue)
{
	// Get and scale down image to 16x16
	cv::Mat scaledDownImage = parseImage(imagePath);

	// Convert image to pixel array
	pixelToValue(scaledDownImage, pixelValue);
}

void parseDataset(std::string datasetPath, int sampleNumber, std::string outputfile)
{
	fstream file(outputfile, ios::out);

	// Foreach sample in dataset
	for (int sample = 1; sample <= sampleNumber; sample++)
	{
		// Foreach character to learn
		for (int character = 0; character < 10; character++)
		{
			// Create absolute path of image indicated in parameter
			std::string imagePath = datasetPath + "\\Sample" + intToString(character + 1) + "\\" + intToString(sample) + ".png";

			// Array containing pixel values of dataset image scaled to 16x16 (256 values)
			int pixelValue[256];
			imageToPixelValue(imagePath, pixelValue);

			// Write pixel array to stream
			writePixelValue(file, pixelValue, character);
		}
	}

	file.close();
}

void read_dataset(char *filename, cv::Mat &data, cv::Mat &classes, int total_samples)
{

	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename, "r");

	//read each row of the csv file
	for (int row = 0; row < total_samples; row++)
	{
		//for each attribute in the row
		for (int col = 0; col <= ATTRIBUTES; col++)
		{
			//if its the pixel value.
			if (col < ATTRIBUTES){

				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row, col) = pixelvalue;

			}//if its the label
			else if (col == ATTRIBUTES){
				//make the value of label column in that row as 1.
				fscanf(inputfile, "%i", &label);
				classes.at<float>(row, label) = 1.0;

			}
		}
	}

	fclose(inputfile);
}

void loadDataToMat(int pixelArray[], cv::Mat &data)
{
	for (int col = 0; col < ATTRIBUTES; ++col)
	{
		data.at<float>(0, col) = (float) pixelArray[col];
	}
}


int _tmain(int argc, _TCHAR* argv[])
{
	size_t size;

	std::string datasetPath = "Z:\\Documents\\Modules HE-Arc\\3252 - Imagerie numérique\\3252.2 Traitement d'image\\Captch-it\\CaptchIt\\Dataset";
	string trainingPath = datasetPath + "\\trainingset.txt";
	string testPath = datasetPath + "\\testset.txt";
	string networkPath = datasetPath + "\\param.xml";
	/*
	 * READ TRAINING SET
	 * Create file with all training set
	 */
	cout << "Reading the training set......\n";
	//parseDataset(datasetPath, 305, datasetPath + "\\trainingset.txt");

	/*
	 * READ TEST SET
	 * Create file with all test set
	 */
	cout << "Reading the test set.........\n";
	//parseDataset(datasetPath, 130, datasetPath + "\\testset.txt");

	/*
	 * TRAIN NEURAL NETWORK
	 */

	//matrix to hold the training sample
	cv::Mat training_set(TRAINING_SAMPLES, ATTRIBUTES, CV_32F);
	//matrix to hold the labels of each taining sample
	cv::Mat training_set_classifications(TRAINING_SAMPLES, CLASSES, CV_32F);
	//matric to hold the test samples
	cv::Mat test_set(TEST_SAMPLES, ATTRIBUTES, CV_32F);
	//matrix to hold the test labels.
	cv::Mat test_set_classifications(TEST_SAMPLES, CLASSES, CV_32F);

	//
	cv::Mat classificationResult(1, CLASSES, CV_32F);
	//load the training and test data sets.
	read_dataset((char*)trainingPath.c_str(), training_set, training_set_classifications, TRAINING_SAMPLES);
	read_dataset((char*)testPath.c_str(), test_set, test_set_classifications, TEST_SAMPLES);

	// define the structure for the neural network (MLP)
	// The neural network has 3 layers.
	// - one input node per attribute in a sample so 256 input nodes
	// - 16 hidden nodes
	// - 10 output node, one for each class.

	cv::Mat layers(3, 1, CV_32S);
	layers.at<int>(0, 0) = ATTRIBUTES;//input layer
	layers.at<int>(1, 0) = 16;//hidden layer
	layers.at<int>(2, 0) = CLASSES;//output layer

	//create the neural network.
	//for more details check http://docs.opencv.org/modules/ml/doc/neural_networks.html
	CvANN_MLP nnetwork(layers, CvANN_MLP::SIGMOID_SYM, 0.6, 1);

	CvANN_MLP_TrainParams params(

		// terminate the training after either 1000
		// iterations or a very small change in the
		// network wieghts below the specified value
		cvTermCriteria(CV_TERMCRIT_ITER + CV_TERMCRIT_EPS, 1000, 0.000001),
		// use backpropogation for training
		CvANN_MLP_TrainParams::BACKPROP,
		// co-efficents for backpropogation training
		// recommended values taken from http://docs.opencv.org/modules/ml/doc/neural_networks.html#cvann-mlp-trainparams
		0.1,
		0.1);

	// train the neural network (using training data)

	printf("\nUsing training dataset\n");
	int iterations = nnetwork.train(training_set, training_set_classifications, cv::Mat(), cv::Mat(), params);
	printf("Training iterations: %i\n\n", iterations);

	// Save the model generated into an xml file.
	CvFileStorage* storage = cvOpenFileStorage((char*) networkPath.c_str(), 0, CV_STORAGE_WRITE);
	nnetwork.write(storage, "DigitOCR");
	cvReleaseFileStorage(&storage);

	// Test the generated model with the test samples.
	cv::Mat test_sample;
	//count of correct classifications
	int correct_class = 0;
	//count of wrong classifications
	int wrong_class = 0;

	//classification matrix gives the count of classes to which the samples were classified.
	int classification_matrix[CLASSES][CLASSES] = { {} };

	// for each sample in the test set.
	for (int tsample = 0; tsample < TEST_SAMPLES; tsample++) {

		// extract the sample

		test_sample = test_set.row(tsample);

		//try to predict its class

		nnetwork.predict(test_sample, classificationResult);
		// The classification result matrix holds weightage  of each class.
		// we take the class with the highest weightage as the resultant class

		// find the class with maximum weightage.
		int maxIndex = 0;
		float value = 0.0f;
		float maxValue = classificationResult.at<float>(0, 0);
		for (int index = 1; index<CLASSES; index++)
		{
			value = classificationResult.at<float>(0, index);
			if (value>maxValue)
			{
				maxValue = value;
				maxIndex = index;

			}
		}

		printf("Testing Sample %i -> class result (digit %d)\n", tsample, maxIndex);

		//Now compare the predicted class to the actural class. if the prediction is correct then\
		            //test_set_classifications[tsample][ maxIndex] should be 1.
		//if the classification is wrong, note that.
		if (test_set_classifications.at<float>(tsample, maxIndex) != 1.0f)
		{
			// if they differ more than floating point error => wrong class

			wrong_class++;

			//find the actual label 'class_index'
			for (int class_index = 0; class_index<CLASSES; class_index++)
			{
				if (test_set_classifications.at<float>(tsample, class_index) == 1.0f)
				{

					classification_matrix[class_index][maxIndex]++;// A class_index sample was wrongly classified as maxindex.
					break;
				}
			}

		}
		else {

			// otherwise correct

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

	/*
	 * TEST NEURAL NETWORK
	 */
	for (int charIndex = 1; charIndex < 11; ++charIndex)
	{
		cout << "--- " << charIndex << " --------------------------------" << endl;

		//read the model from the XML file and create the neural network.
		std::string neuralNetworkParameters = datasetPath + "\\param.xml";
		CvANN_MLP nnetwork;
		int imageDataPixelValue[256];
		imageToPixelValue(datasetPath + "\\Sample" + intToString(charIndex) + "\\1.png", imageDataPixelValue);
		CvFileStorage* storage = cvOpenFileStorage((char*) neuralNetworkParameters.c_str(), 0, CV_STORAGE_READ);
		CvFileNode *n = cvGetFileNodeByName(storage, 0, "DigitOCR");
		nnetwork.read(storage, n);
		cvReleaseFileStorage(&storage);

		cv::Mat data(1, ATTRIBUTES, CV_32F);

		loadDataToMat(imageDataPixelValue, data);

		// Generate cv::Mat data(1,ATTRIBUTES,CV_32F) which will contain the pixel data for the digit to be recognized
		//cv::Mat data(1, ATTRIBUTES, CV_32F, &imageDataPixelValue);

		int maxIndex = 0;
		cv::Mat classOut(1, CLASSES, CV_32F);

		// Prediction
		nnetwork.predict(data, classOut);
		float value;
		float maxValue = classOut.at<float>(0, 0);
		for (int index = 1; index<CLASSES; index++)
		{
			value = classOut.at<float>(0, index);
			//cout << "Classe:" << index << " - Probability : " << value << endl;
			if (value>maxValue)
			{
				maxValue = value;
				maxIndex = index;
			}
		}

		cout << "Classe:" << maxIndex << " - Probability : " << maxValue << endl;
	}
	
	system("PAUSE");
	return 0;
}

