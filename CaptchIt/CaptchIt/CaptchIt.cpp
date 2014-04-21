#include "stdafx.h"
#include <stdlib.h>
#include <string.h>
#include <fstream>
#include <io.h>

#include "CvNeuralNetwork.h"

using namespace cv;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	string datasetPath = "Z:\\Documents\\Modules HE-Arc\\3252 - Imagerie numérique\\3252.2 Traitement d'image\\Captch-it\\CaptchIt\\Dataset";
	string trainingPath = datasetPath + "\\trainingset.txt";
	string testPath = datasetPath + "\\testset.txt";
	string networkPath = datasetPath + "\\param.xml";

	cout << "Writing the training set ..." << endl;
	//CvTools::writeDataset(datasetPath, 305, datasetPath + "\\trainingset.txt");

	cout << "Writing the test set ..." << endl;
	//CvTools::writeDataset(datasetPath, 130, datasetPath + "\\testset.txt");

	CvNeuralNetwork neuralNetwork(datasetPath, trainingPath, testPath, networkPath);

	neuralNetwork.testParameters();

	for (int charIndex = 1; charIndex < 11; ++charIndex)
	{
		string filename = datasetPath + "\\Sample" + CvTools::intToString(charIndex) + "\\1.png";

		cout << "-------------------- " << charIndex << " --------------------" << endl;

		int maxIndex = neuralNetwork.predictClass(filename);

		cout << "Classe:" << maxIndex << endl;
	}
	
	system("PAUSE");
	return 0;
}

