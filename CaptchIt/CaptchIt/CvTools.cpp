#include "stdafx.h"
#include "CvTools.h"
#include <sys/stat.h>

#pragma warning (disable : 4996)
#include <stdio.h>

CvTools::CvTools()
{
}

CvTools::~CvTools()
{
}

bool CvTools::fileExists(const string& name) 
{
	// Fatest way to check if a file exists
	// http://stackoverflow.com/questions/12774207/fastest-way-to-check-if-a-file-exist-using-standard-c-c11-c

	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void CvTools::scaleDownImage(cv::Mat &originalImg, cv::Mat &scaledDownImage)
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

void CvTools::cropImage(cv::Mat &originalImage, cv::Mat &croppedImage)
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

void CvTools::pixelToValue(cv::Mat &img, int pixelArray[])
{
	int i = 0;
	for (int x = 0; x < 16; ++x)
	{
		for (int y = 0; y < 16; ++y)
		{
			// Image is thresholded :
			// Value = 1 if white pixel (255), 0 otherwise (black pixel)
			pixelArray[i] = (img.at<uchar>(x, y) == 255) ? 1 : 0;
			++i;
		}
	}
}

string CvTools::intToString(int number)
{
	stringstream ss;
	ss << number;
	return ss.str();
}

string CvTools::pixelValueToString(int pixelArray[])
{
	string finalString;

	for (int i = 0; i < CvNeuralNetwork::ATTRIBUTES; ++i)
	{
		finalString += CvTools::intToString(pixelArray[i]);
	}

	return finalString;
}

Mat CvTools::parseImage(std::string imagePath)
{
	// Load image in a matrix
	Mat img = cv::imread(imagePath, 0);
	Mat output;

	// Remove noise
	GaussianBlur(img, output, cv::Size(5, 5), 0);

	// Transform to binary image
	threshold(output, output, 50, 255, 0);

	// Variable to store scaled image
	Mat scaledDownImage(16, 16, CV_8U, cv::Scalar(0));

	// Crop image
	CvTools::cropImage(output, output);

	// Scale down image to 16x16
	CvTools::scaleDownImage(output, scaledDownImage);

	return scaledDownImage;
}

void CvTools::writePixelValue(fstream &file, int* pixelValue, int character)
{
	for (int pixelIndex = 0; pixelIndex < 256; ++pixelIndex)
	{
		file << pixelValue[pixelIndex] << ",";
	}

	// Write the label at the end of the data
	file << character << "\n";
}

void CvTools::imageToPixelValue(std::string imagePath, int* pixelValue)
{
	// Get and scale down image to 16x16
	cv::Mat scaledDownImage = CvTools::parseImage(imagePath);

	// Convert image to pixel array
	CvTools::pixelToValue(scaledDownImage, pixelValue);
}

void CvTools::writeDataset(std::string datasetPath, int sampleNumber, std::string outputfile)
{
	fstream file(outputfile, ios::out);

	// Foreach sample in dataset
	for (int sample = 1; sample <= sampleNumber; sample++)
	{
		// Foreach character to learn
		for (int character = 0; character < 10; character++)
		{
			// Create absolute path of image indicated in parameter
			std::string imagePath = datasetPath + "\\Sample" + CvTools::intToString(character + 1) + "\\" + CvTools::intToString(sample) + ".png";

			// Array containing pixel values of dataset image scaled to 16x16 (256 values)
			int pixelValue[256];
			CvTools::imageToPixelValue(imagePath, pixelValue);

			// Write pixel array to stream
			CvTools::writePixelValue(file, pixelValue, character);
		}
	}

	file.close();
}

void CvTools::readDataset(char *filename, cv::Mat &data, cv::Mat &classes, int sampleNumber)
{

	int label;
	float pixelvalue;
	//open the file
	FILE* inputfile = fopen(filename, "r");

	//read each row of the csv file
	for (int row = 0; row < sampleNumber; row++)
	{
		//for each attribute in the row
		for (int col = 0; col <= CvNeuralNetwork::ATTRIBUTES; col++)
		{
			// Pixel value
			if (col < CvNeuralNetwork::ATTRIBUTES)
			{
				fscanf(inputfile, "%f,", &pixelvalue);
				data.at<float>(row, col) = pixelvalue;
			}
			// Last column = Corresponding character (class)
			// Save class to show result
			else if (col == CvNeuralNetwork::ATTRIBUTES)
			{
				fscanf(inputfile, "%i", &label);
				classes.at<float>(row, label) = 1.0;
			}
		}
	}

	fclose(inputfile);
}

void CvTools::loadDataToMat(int pixelArray[], cv::Mat &data)
{
	for (int col = 0; col < CvNeuralNetwork::ATTRIBUTES; ++col)
	{
		data.at<float>(0, col) = (float)pixelArray[col];
	}
}

int CvTools::searchMaxWeight(Mat &classificationResult)
{
	int maxIndex = 0;
	float value = 0.0f;
	float maxValue = classificationResult.at<float>(0, 0);

	for (int index = 1; index < CvNeuralNetwork::CLASSES; ++index)
	{
		value = classificationResult.at<float>(0, index);

		if (value > maxValue)
		{
			maxValue = value;
			maxIndex = index;
		}
	}

	return maxIndex;
}