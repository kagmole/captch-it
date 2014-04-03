#include "stdafx.h"

using namespace cv;
using namespace std;

#include <stdio.h>
#include <iomanip>
#include <time.h>

const char *VENTANA = "Blob detection";



int _tmain(int argc, _TCHAR* argv[])
{
	//Mat easyCaptchaImage = imread("images/easy.png", CV_LOAD_IMAGE_UNCHANGED);

	IplImage* threshold;
	IplImage* imagen;
	IplImage* imagen_color;
	IplImage* smooth;

	IplImage* open_morf;
	IplImage* blobResult;
	IplImage* img_contornos;
	CvMoments moments;
	CvHuMoments humoments;
	CvSeq* contour;
	CvSeq* contourLow;
	srand(time(NULL));
	char* filename = "images/easy.png";
	//load image  in gray level
	imagen = cvLoadImage(filename, 0);
	//load image in RGB
	imagen_color = cvLoadImage(filename, 1);

	smooth = cvCreateImage(cvSize(imagen->width, imagen->height), IPL_DEPTH_8U, 1);
	threshold = cvCreateImage(cvSize(imagen->width, imagen->height), IPL_DEPTH_8U, 1);
	open_morf = cvCreateImage(cvSize(imagen->width, imagen->height), IPL_DEPTH_8U, 1);

	//Init variables for countours
	contour = 0;
	contourLow = 0;
	//Create storage needed for contour detection
	CvMemStorage* storage = cvCreateMemStorage(0);

	//Create window
	cvNamedWindow(VENTANA, 0);

	//Smooth image
	cvSmooth(imagen, smooth, CV_GAUSSIAN, 3, 0, 0, 0);

	CvScalar avg;
	CvScalar avgStd;
	cvAvgSdv(smooth, &avg, &avgStd, NULL);
	printf("Avg: %f\nStd: %f\n", avg.val[0], avgStd.val[0]);
	//threshold image
	cvThreshold(smooth, threshold, (int)avg.val[0] - 7 * (int)(avgStd.val[0] / 8), 255, CV_THRESH_BINARY_INV);
	//Morfologic filters
	cvErode(threshold, open_morf, NULL, 1);
	cvDilate(open_morf, open_morf, NULL, 1);
	//Duplicate image for countour
	img_contornos = cvCloneImage(open_morf);

	//Search countours in preprocesed image
	cvFindContours(img_contornos, storage, &contour, sizeof(CvContour),
		CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE, cvPoint(0, 0));
	//Optimize contours, reduce points
	contourLow = cvApproxPoly(contour, sizeof(CvContour), storage, CV_POLY_APPROX_DP, 1, 1);

	//Show to user the 7 hu moments (features detected) we can use this for clasification
	printf("hu1\t\thu2\t\thu3\t\thu4\t\thu5\t\thu6\t\thu7\n");

	string folderName = "segments";
	string folderCreateCommand = "mkdir " + folderName;
	system(folderCreateCommand.c_str());

	//For each contour found
	for (int i=0; contourLow != 0; contourLow = contourLow->h_next)
	{
		CvScalar color = CV_RGB(rand() & 200, rand() & 200, rand() & 200);
		//We can draw the contour of object
		//cvDrawContours( imagen_color, contourLow, color, color, -1, 0, 8, cvPoint(0,0) );
		//Or detect bounding rect of contour	
		CvRect rect;
		CvPoint pt1, pt2;
		rect = cvBoundingRect(contourLow, NULL);
		pt1.x = rect.x;
		pt2.x = (rect.x + rect.width);
		pt1.y = rect.y;
		pt2.y = (rect.y + rect.height);

		Mat toCrop(imagen_color);
		Mat crop = toCrop(rect);

		ostringstream convert;   // stream used for the conversion
		convert << folderName << "/segment" << setfill('0') << setw(3) << i++ << ".png";      // insert the textual representation of 'Number' in the characters in the stream
		//imshow(convert.str(), crop);

		imwrite(convert.str(), crop);

		//cvRectangle(imagen_color, pt1, pt2, color, 1, 8, 0);

		//For calculate objects features we can use hu moments.
		//this calculate 7 features that are independent of position, size and orientation

		//First calculate object moments
		cvMoments(contourLow, &moments, 0);
		//Now calculate hu moments
		cvGetHuMoments(&moments, &humoments);

		//Show result.
		//printf("%f\t%f\t%f\t%f\t%f\t%f\t%f\n", humoments.hu1, humoments.hu2, humoments.hu3, humoments.hu4, humoments.hu5, humoments.hu6, humoments.hu7);
	}

	cvShowImage(VENTANA, imagen_color);
	cvShowImage(VENTANA, imagen_color);
	cvShowImage("asdasd", imagen_color);
	cvShowImage(VENTANA, imagen_color);

	////Main Loop
	for (;;)
	{
		int c;
		c = cvWaitKey(10);
		if ((char)c == 27)
			break;
		else if ((char)c == '3')
			cvShowImage(VENTANA, threshold);
		else if ((char)c == '1')
			cvShowImage(VENTANA, imagen);
		else if ((char)c == '2')
			cvShowImage(VENTANA, smooth);
		else if ((char)c == '4')
			cvShowImage(VENTANA, open_morf);
		else if ((char)c == 'r')
			cvShowImage(VENTANA, imagen_color);
	}

	cvDestroyWindow(VENTANA);
	
	return 0;
}

