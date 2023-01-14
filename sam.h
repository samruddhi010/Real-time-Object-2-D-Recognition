/***********************************Project 3: Real time 2D object Detection**********************************
 Created by Samruddhi Raut.
  
 header file - descriptions of functions are provided in the features.cpp

    File Name: sam.h
    This file contains code for below headers:
     1.Customise threshold
     2.Threshold value
     3.Erode
     4.Dilate
     5.Grassfire transform
     6.Connected component label


    
 
*/

#pragma once
#include <opencv2/opencv.hpp>


using namespace cv;
using namespace std;

int thresholdVal(cv::Mat &src);

int thresholdImage(cv::Mat &src, cv::Mat &dst); 

int grassfireTransform(Mat& target, Mat& local, int c, int connectivity);

int erode(cv::Mat &src, cv::Mat &dst);

int dilate(cv::Mat &src, cv::Mat &dst);

std::vector<int> connectedComponentLabel(cv::Mat &src, cv::Mat &dst, cv::Mat &vis);






