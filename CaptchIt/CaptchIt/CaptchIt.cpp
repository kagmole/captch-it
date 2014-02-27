#include "stdafx.h"

using namespace cv;
using namespace std;

int _tmain(int argc, _TCHAR* argv[])
{
	Mat img = imread("MyPic.JPG", CV_LOAD_IMAGE_UNCHANGED);
	
	if (img.empty()) {
		cout << "Error: Image cannot be loaded!" << endl;
		return -1;
	}
	
	namedWindow("MyWindow", CV_WINDOW_AUTOSIZE);
	imshow("MyWindow", img);
	
	waitKey(0);
	
	destroyWindow("MyWindow");
	
	return 0;
}

