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
#include <stdio.h>

#define ATTRIBUTES 256  //Number of pixels per sample.16X16
#define CLASSES 10                  //Number of distinct labels.

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


int _tmain(int argc, _TCHAR* argv[])
{
	size_t size;

	std::string datasetPath = "Z:\\Documents\\Modules HE-Arc\\3252 - Imagerie numérique\\3252.2 Traitement d'image\\Captch-it\\CaptchIt\\Dataset";

	system("PAUSE");

	cout << "Reading the training set......\n";
	//parseDataset(datasetPath, 305, datasetPath + "\\trainingset.txt");


	system("PAUSE");

	cout << "Reading the test set.........\n";
	//parseDataset(datasetPath, 130, datasetPath + "\\testset.txt");

	system("PAUSE");

	//read the model from the XML file and create the neural network.
	std::string neuralNetworkParameters = datasetPath + "\\param.xml";
	CvANN_MLP nnetwork;
	int imageDataPixelValue[256];
	imageToPixelValue(datasetPath + "\\Sample5\\162.png", imageDataPixelValue);
	CvFileStorage* storage = cvOpenFileStorage(neuralNetworkParameters.c_str(), 0, CV_STORAGE_READ);
	CvFileNode *n = cvGetFileNodeByName(storage, 0, "DigitOCR");
	nnetwork.read(storage, n);
	cvReleaseFileStorage(&storage);

	//string stringPixelValue = pixelValueToString(imageDataPixelValue);

	cv::Mat data(1, ATTRIBUTES, CV_32F, &imageDataPixelValue);

	// Generate cv::Mat data(1,ATTRIBUTES,CV_32S) which will contain the pixel data for the digit to be recognized

	int maxIndex = 0;
	cv::Mat classOut(1, CLASSES, CV_32F); // CV_32F
	//prediction
	nnetwork.predict(data, classOut);
	float value;
	float maxValue = classOut.at<float>(0, 0);
	for (int index = 1; index<CLASSES; index++)
	{
		value = classOut.at<float>(0, index);
		if (value>maxValue)
		{
			maxValue = value;
			maxIndex = index;
		}
	}

	cout << maxValue << endl;
	cout << maxIndex << endl;
	
	system("PAUSE");
	return 0;
}

