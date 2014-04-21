#pragma once

#include "opencv2/opencv.hpp"
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <fstream>

#include "CvNeuralNetwork.h"

using namespace std;
using namespace cv;

class CvTools
{
public:

	static bool fileExists(const string& name);
	static void scaleDownImage(Mat &originalImg, Mat &scaledDownImage);
	static void cropImage(Mat &originalImage, Mat &croppedImage);
	static void pixelToValue(Mat &img, int pixelArray[]);
	static string intToString(int number);
	static string pixelValueToString(int pixelArray[]);
	static Mat parseImage(string imagePath);
	static void writePixelValue(fstream &file, int* pixelValue, int character);
	static void imageToPixelValue(string imagePath, int* pixelValue);
	static void writeDataset(std::string datasetPath, int sampleNumber, std::string outputfile);
	static void readDataset(char *filename, Mat &data, Mat &classes, int sampleNumber);
	static void loadDataToMat(int pixelArray[], Mat &data);
	static int searchMaxWeight(Mat &classificationResult);

private:
	CvTools();
	~CvTools();
};

