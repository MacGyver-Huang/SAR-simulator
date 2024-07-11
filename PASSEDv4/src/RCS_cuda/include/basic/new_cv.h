//
//  new_cv.h
//  SARSFCWPro
//
//  Created by Chiang Steve on 6/5/12.
//  Copyright (c) 2012 NCU. All rights reserved.
//

#ifndef SARSFCWPro_new_cv_h
#define SARSFCWPro_new_cv_h

#include "d1.h"
#include "d2.h"
#include "mat.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>
#include <opencv/ml.h>
#include <opencv2/opencv.hpp>

using namespace cv;

namespace new_cv {
	template<typename T> D2<T> Scale0to255(const D2<T>& data);
	void aimage(const char* filename);
	template<typename T> void aimage(const D2<T>& data);
	template<typename T> void Plot(const D1<T>& Y);
	template<typename T> void Plot(const D1<T>& X, const D1<T>& Y, 
								   CvScalar* LineColor, CvScalar* BgColor, int* LineThick);
	void TruncateImage(Mat& img, const float truc_head, const float truc_tail);
	void Rot90(Mat &matImage, int rotflag);
	// Color table
	namespace lct {
		template<typename T> D1<int> jet(const T in);
		Mat ENVI(Mat& im_gray);
	}
	// IO
	template<typename T> void WriteBMP(const D2<T>& data, const char* filename);
	void WriteImage(const Mat& mesh, const string file_render, bool IsROT90=false);
	void WriteImagePesudoSAR(const Mat& mesh, const string file_render, const string file_shader);
}

template<typename T>
D2<T> new_cv::Scale0to255(const D2<T>& data){
	//
	// Scaling from 0 ~ 255
	//
	long m = data.GetM();
	long n = data.GetN();
	// find max & min values
	T min_val = mat::min(data);
	T max_val = mat::max(data);
	T sub = max_val - min_val;
	
	// scaling
	D2<T> data_export(m,n);
	for(long j=0L;j<m;++j){
		for(long i=0L;i<n;++i){
			// scaling & flipud
			data_export[m-j-1L][i] = (data[j][i] - min_val)/sub * 255;
		}
	}
	return data_export;
}

void new_cv::aimage(const char* filename){
	//get the image from the directed path  
    IplImage* img = cvLoadImage(filename, 1);
	
    //create a window to display the image  
    cvNamedWindow("picture", 1);
    //show the image in the window  
    cvShowImage("picture", img);  
    //wait for the user to hit a key  
    cvWaitKey(0);
    //delete the image and window  
    cvReleaseImage(&img);
    cvDestroyWindow("picture");
}


template<typename T>
void new_cv::aimage(const D2<T>& data){
	long m = data.GetM();
	long n = data.GetN();
	CvSize ImageSize1 = cvSize((int)n,(int)m);
	IplImage* Image1 = cvCreateImage(ImageSize1,IPL_DEPTH_8U,3);
	
	// Set 0~255
	D2<T> data0to255 = new_cv::Scale0to255(data);
	//cout<<mat::max(data0to255)<<endl;
	//cout<<mat::min(data0to255)<<endl;
	D1<int> RGB(3);
	int tmp;
	// Set data
	for(int i=0;i<Image1->height;i++){
        for(int j=0;j<Image1->widthStep-3;j=j+3){
//			if( (*(data0to255.GetPtr()+(j/3)+i*n)) == 255 ){
//				cout<<123<<endl;
//			}
			//RGB = new_cv::lct::jet( *(data0to255.GetPtr()+(j/3)+i*n) );
			tmp = *(data0to255.GetPtr()+(j/3)+i*n);
			RGB[0] = tmp; RGB[1] = tmp; RGB[2] = tmp;
            Image1->imageData[i*Image1->widthStep+j]   = RGB[2]; // B
            Image1->imageData[i*Image1->widthStep+j+1] = RGB[1]; // G
            Image1->imageData[i*Image1->widthStep+j+2] = RGB[0]; // R
        }
    }
	
	cvNamedWindow("aimage",1);
	cvShowImage("aimage",Image1);
	cvWaitKey(0);
}

//template<typename T>
//void new_cv::Plot(const D1<T>& Y){
//	int WinSz[2] = {600,300};
//
//	CvSize ImageSize1 = cvSize(WinSz[0],WinSz[1]);
//    IplImage* Image1 = cvCreateImage(ImageSize1,IPL_DEPTH_8U,3);
//
//	CvScalar bgcolor = CV_RGB(50,50,50);
//	CvScalar linecolor = CV_RGB(255,0,0);
//
//	//CvScalar bg_color = CV_RGB(50, 50, 50);
//	for(int i=0;i<Image1->height;i++){
//        for(int j=0;j<Image1->widthStep;j=j+3){
//            Image1->imageData[i*Image1->widthStep+j]=bgcolor.val[0];
//            Image1->imageData[i*Image1->widthStep+j+1]=bgcolor.val[1];
//            Image1->imageData[i*Image1->widthStep+j+2]=bgcolor.val[2];
//        }
//    }
//
//	long num = Y.GetNum();
//	D1<T> X(num);
//	for(long i=0;i<num;++i){ X[i] = T(i); }
//
//
//	int marginx = int(double(WinSz[0])*0.1);
//	int marginy = int(double(WinSz[1])*0.1);
//
//	int Xrg[2] = {marginx, WinSz[0] - marginx};
//	int Yrg[2] = {marginy, WinSz[1] - marginy};
//
//
//	T minx = mat::min(X);
//	T maxx = mat::max(X);
//	T miny = mat::min(Y);
//	T maxy = mat::max(Y);
//
//	CvPoint* Points = new CvPoint[num];
//	for(long i=0L;i<num;++i){
//		Points[i].x = (Xrg[1]-Xrg[0])/(maxx-minx) * (X[i]-minx) + Xrg[0];
//		Points[i].y = (Yrg[1]-Yrg[0])/(maxy-miny) * (Y[i]-miny) + Yrg[0];
//	}
//
//
//    int Thickness=1;
//
//
//    int Shift=0;
//	for(long i=0L;i<num-1;++i){
//		cvLine(Image1,Points[i],Points[i+1],linecolor,Thickness,CV_AA,Shift);
//	}
//
//	// Axis-X
//	cvLine(Image1,cvPoint(Xrg[0],(Yrg[0]+Yrg[1])/2),cvPoint(Xrg[1],(Yrg[0]+Yrg[1])/2),
//		   CV_RGB(255,255,255),1,CV_AA,Shift);
//
//	// Axis-Y
//	cvLine(Image1,cvPoint(Xrg[0],Yrg[0]),cvPoint(Xrg[0],Yrg[1]),
//		   CV_RGB(255,255,255),1,CV_AA,Shift);
//
//	cvFlip(Image1, NULL, 1);
//    cvNamedWindow("Plot",1);
//    cvShowImage("Plot",Image1);
//
//	cvWaitKey(0);
//
//
//
//	// clean up
//	delete[] Points;
//}

//template<typename T>
//void new_cv::Plot(const D1<T>& X, const D1<T>& Y,
//				  CvScalar* LineColor, CvScalar* BgColor, int* LineThick){
//	int WinSz[2] = {600,300};
//
//	CvSize ImageSize1 = cvSize(WinSz[0],WinSz[1]);
//    IplImage* Image1 = cvCreateImage(ImageSize1,IPL_DEPTH_8U,3);
//
//	cv::Scalar bgcolor = CV_RGB(50.,50.,50.);
//	if(BgColor != NULL){ bgcolor = (*BgColor); }
//	cv::Scalar linecolor = CV_RGB(255,0,0);
//	if(LineColor != NULL){ linecolor = (*LineColor); }
//
//	//CvScalar bg_color = CV_RGB(50, 50, 50);
//	for(int i=0;i<Image1->height;i++){
//        for(int j=0;j<Image1->widthStep;j=j+3){
//            Image1->imageData[i*Image1->widthStep+j]=bgcolor.val[0];
//            Image1->imageData[i*Image1->widthStep+j+1]=bgcolor.val[1];
//            Image1->imageData[i*Image1->widthStep+j+2]=bgcolor.val[2];
//        }
//    }
//
//	long num = X.GetNum();
//
//
//	int marginx = int(double(WinSz[0])*0.1);
//	int marginy = int(double(WinSz[1])*0.1);
//
//	int Xrg[2] = {marginx, WinSz[0] - marginx};
//	int Yrg[2] = {marginy, WinSz[1] - marginy};
//
//
//	T minx = mat::min(X);
//	T maxx = mat::max(X);
//	T miny = mat::min(Y);
//	T maxy = mat::max(Y);
//
//	CvPoint* Points = new CvPoint[num];
//	for(long i=0L;i<num;++i){
//		Points[i].x = (Xrg[1]-Xrg[0])/(maxx-minx) * (X[i]-minx) + Xrg[0];
//		Points[i].y = (Yrg[1]-Yrg[0])/(maxy-miny) * (Y[i]-miny) + Yrg[0];
//	}
//
//
//    int Thickness=1;
//	if(LineThick != NULL){ Thickness = (*LineThick); }
//
//
//    int Shift=0;
//	for(long i=0L;i<num-1;++i){
//		cv::line(Image1,Points[i],Points[i+1],linecolor,Thickness,CV_AA,Shift);
//	}
//
//	// Axis-X
//	cvLine(Image1,cvPoint(Xrg[0],(Yrg[0]+Yrg[1])/2),cvPoint(Xrg[1],(Yrg[0]+Yrg[1])/2),
//		   CV_RGB(255,255,255),1,CV_AA,Shift);
//
//	// Axis-Y
//	cvLine(Image1,cvPoint(Xrg[0],Yrg[0]),cvPoint(Xrg[0],Yrg[1]),
//		   CV_RGB(255,255,255),1,CV_AA,Shift);
//
//    cvNamedWindow("Plot",1);
//    cvShowImage("Plot",Image1);
//
//	cvWaitKey(0);
//
//
//
//	// clean up
//	delete[] Points;
//}

template<typename T>
void new_cv::WriteBMP(const D2<T>& data, const char* filename){
	
	int DEPTH = -99;
	if( typeid(T) == typeid(double(1)) ){ DEPTH = IPL_DEPTH_64F; }
	if( typeid(T) == typeid(float(1)) ) { DEPTH = IPL_DEPTH_32F; }
	
	if(DEPTH != -99){
		long m = data.GetM();
		long n = data.GetN();
		IplImage *image;
		CvSize size = cvSize((int)n,(int)m);// stand for D2<T>(_m,_n)
		image = cvCreateImageHeader( size, DEPTH, 1 );
		// Scaling from 0 ~ 255
		D2<T> data_export = new_cv::Scale0to255(data);
		// Data is column major (diff. D2<T> row major)
		cvSetData( image, data_export.GetPtr(), image->widthStep );
		cvSaveImage( filename, image );
	}
}

// for pseudo SAR images
void new_cv::TruncateImage(Mat& img, const float truc_head, const float truc_tail){
	for(int i=0;i<img.rows;++i){
		for(int j=0;j<img.cols;++j){
			float val = img.at<float>(i,j);
			if(val < truc_head){ img.at<float>(i,j) = truc_head; }
			if(val > truc_tail){ img.at<float>(i,j) = truc_tail; }
		}
	}
}

void new_cv::Rot90(Mat &matImage, int rotflag){
	//1=CW, 2=CCW, 3=180
	if (rotflag == 1){
		transpose(matImage, matImage);
		flip(matImage, matImage,1); //transpose+flip(1)=CW
	} else if (rotflag == 2) {
		transpose(matImage, matImage);
		flip(matImage, matImage,0); //transpose+flip(0)=CCW
	} else if (rotflag ==3){
		flip(matImage, matImage,-1);    //flip(-1)=180
	} else if (rotflag != 0){ //if not 0,1,2,3:
		cerr<<"ERROR::Unknown rotation flag(" << rotflag << ")" << endl;
	}
}

void new_cv::WriteImage(const Mat& mesh, const string file_render, bool IsROT90){
	// Output image file with ENVI color table (jpg)
	//
	// Truncate image
	//
	double MinValue;
	double MaxValue;
	minMaxLoc(mesh, &MinValue, &MaxValue);
	//
	// Scale to 0~255
	//
	Mat Scale;
	double alpha = 255./(MaxValue-MinValue);
	double beta  = -MinValue*alpha;
	mesh.convertTo(Scale, CV_8U, alpha, beta);
	//
	// apply color map
	//
	//	Mat out_img;
	//	applyColorMap(Scale, out_img, COLORMAP_JET);
	cvtColor(Scale.clone(), Scale, COLOR_GRAY2BGR);
	Mat out_img = lct::ENVI(Scale);
	//
	// Save jpg
	//
	if(IsROT90){
		Rot90(out_img, 2);
	}
	flip(out_img, out_img, 0);
	imwrite(file_render.c_str(), out_img);
}

void new_cv::WriteImagePesudoSAR(const Mat& mesh, const string file_render, const string file_shader){

	// Get size
	Size sz = mesh.size();
	int width  = sz.width;
	int height = sz.height;

	// Output image file with ENVI color table (jpg)
	WriteImage(mesh, file_render);

	// Find the max and min(non-zero) value
	float minval = 1E+30, maxval = 1E-30;
	for(int i=0;i<height;++i){
		for(int j=0;j<width;++j){
			float val = mesh.at<float>(i,j);
			if(val > maxval){ maxval = val; }
			if(val < minval && val != 0){ minval = val; }
		}
	}
	//	double minval, maxval;
	//	cv::minMaxLoc(mesh, &minval, &maxval);

//	cout<<"Min Value = "<<minval<<endl;
//	cout<<"Max Value = "<<maxval<<endl;

	// Calcuate pesudo SAR amplitude
	float range[] = { (float)minval, (float)maxval };
	const float *ranges = { range };


	int histSize = height; // bin size
	Mat shad(height, width, CV_32F);

	// height = 600
	// width  = 800

	for(int i=0;i<width;++i){
		Mat col(height, 1, CV_32F);
		mesh.col(i).copyTo(col);

		Mat hist;
		calcHist(&col, 1, 0, Mat(), hist, 1, &histSize, &ranges, true, false);

		hist.copyTo(shad.col(i));
	}


	// export to jpg
//	flip(shad, shad, 0);
	WriteImage(shad, file_shader, false);
}




//
// namespace lct
//
template<typename T>
D1<int> new_cv::lct::jet(const T in){
	double v = 360./255.*in;
	int s = int(floor(v/60.0));
	D1<int> RGB(3);
	RGB[0] = 0; RGB[1] = 0; RGB[2] = 0;
	if((s >= 0)&&(s <= 1)){ RGB[0] = 255;             RGB[1] = int(255./60.*v); RGB[2] = 0; }
	if((s >= 1)&&(s <= 2)){ RGB[0] = int(255./60.*v); RGB[1] = 0;               RGB[2] = 0; }
	if((s >= 2)&&(s <= 3)){ RGB[0] = 0;               RGB[1] = 0;               RGB[2] = int(255./60.*v); }
	if((s >= 3)&&(s <= 4)){ RGB[0] = 0;               RGB[1] = int(255./60.*v); RGB[2] = 255; }
	if((s >= 4)&&(s <= 5)){ RGB[0] = int(255./60.*v); RGB[1] = 0;               RGB[2] = 255; }
	if((s >= 5)&&(s <= 6)){ RGB[0] = 255;             RGB[1] = 0;               RGB[2] = int(255./60.*v); }
	return RGB;
}

Mat new_cv::lct::ENVI(Mat& im_gray){
	unsigned char r[] = {0,4,9,3,8,2,7,1,6,0,45,50,58,61,64,68,69,72,74,77,79,80,82,83,84,86,87,88,86,87,87,87,85,84,84,84,79,78,77,76,71,70,68,66,60,58,55,46,43,40,36,33,25,21,16,12,4,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,8,12,21,25,29,42,46,51,55,63,67,72,76,80,89,93,97,110,114,119,123,131,135,140,144,153,157,161,165,178,182,187,191,199,203,208,212,221,225,229,242,246,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255};
	unsigned char g[] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,4,16,21,25,29,38,42,46,51,55,63,67,72,84,89,93,97,106,110,114,119,127,131,135,140,152,157,161,165,174,178,182,187,195,199,203,216,220,225,229,233,242,246,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,250,242,238,233,229,221,216,212,199,195,191,187,178,174,170,165,161,153,148,144,131,127,123,119,110,106,102,97,89,85,80,76,63,59,55,51,42,38,34,29,21,17,12,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255};
	unsigned char b[] = {0,3,7,10,14,19,23,28,32,38,43,48,59,63,68,72,77,81,86,91,95,100,104,109,118,122,127,132,136,141,145,150,154,159,163,168,177,182,186,191,195,200,204,209,214,218,223,232,236,241,245,250,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,255,246,242,238,225,220,216,212,203,199,195,191,187,178,174,170,157,152,148,144,135,131,127,123,114,110,106,102,89,84,80,76,67,63,59,55,46,42,38,25,21,16,12,8,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,255};

	Mat channels[] = {Mat(256,1, CV_8U, b), Mat(256,1, CV_8U, g), Mat(256,1, CV_8U, r)};
	Mat lut; // Create a lookup table
	merge(channels, 3, lut);

	Mat im_color;
	LUT(im_gray, lut, im_color);

	return im_color;
}


#endif
