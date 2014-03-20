#include "stdafx.h"

using namespace cv;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	Mat easyCaptchaImage = imread("images/easy.png", CV_LOAD_IMAGE_UNCHANGED);
	
	if (easyCaptchaImage.empty()) {
		cout << "Error: Image cannot be loaded!" << endl;
		cin.get();
		return -1;
	}
	
	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
	imshow("MyWindow", easyCaptchaImage);
	
	waitKey(0);
	
	destroyWindow("MyWindow");
	
	return 0;
}

