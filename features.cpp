/***********************************Project 3: Real time 2D object Detection**********************************
 Created by Samruddhi Raut.
  
    File Name: features.cpp
    This file contains code for below functions:
     1.Customise threshold
     2.Threshold value
     3.Erode
     4.Dilate
     5.Grassfire transform

    Instructions to run the file:
    make
    ./Project_3
 
*/


#include <iostream>
#include <opencv2/opencv.hpp>
#include <vector>
#include <math.h>
#include <opencv2/imgproc.hpp>
#include <fstream>

using namespace cv;
using namespace std;
 /*********************************************Task1************************************************************?
/*
Description: *  This function apply thresholds on input image and return the result.
                takes in the rgb image, converts it to grayscale and then creates a 
                binary image using threshold. The background will be black after the threshold application.
Parameters: src-is a input image matrix
output: results store in dst matrix(uchar type)

*/

int thresholdImage(cv::Mat &src, cv::Mat &dst){
    cv::Mat grayImg(src.rows, src.cols, CV_8UC1, cv::Scalar(0));
    cv::cvtColor(src, grayImg, cv::COLOR_BGR2GRAY);
    for(int i=0; i<grayImg.rows; i++){
        for(int j=0; j<grayImg.cols; j++){
            if ((short)grayImg.at<uchar>(i, j) < 100){
                dst.at<uchar>(i, j) = 255;
            }
        }
    }
    return 0;
}
// This is a function for printing the histogram of pixel values inorder to  manually select threshold values//

int thresholdVal(cv::Mat &src){ 
    int histogram[255] = {0};
    for(int i=0; i<src.rows; i++){
        for(int j=0; j<src.cols; j++){
            histogram[(short)src.at<uchar>(i, j)] += 1;         
        }
    }
    for(int k=0; k<255; k++){
        std::cout<< k + 1 << " : " << histogram[k] <<std::endl;
    }
    return 0;
}

 /*********************************************Task2************************************************************
/*
Description: *  This function erodes the thresholded image and return the result.Performs 4 connect erosion.
Parameters: src-is a input image matrix(uchar type)
output: results store in dst matrix(uchar type)
*/

int erode(cv::Mat &src, cv::Mat &dst){
    
    for(int i=1; i<src.rows-1; i++){
        for(int j=1; j<src.cols-1; j++){
            if( (short)src.at<uchar>(i-1, j) > 250 || (short)src.at<uchar>(i, j-1) > 250 || (short)src.at<uchar>(i, j+1) 
            > 250 ||  (short)src.at<uchar>(i+1, j) > 250){
                dst.at<uchar>(i, j) = 255;
            }else{
                dst.at<uchar>(i, j) = 0;
            }
        }
    }

    return 0;
}

/*
Description: *  This function dilates the eroded image and return the result.Performs 4 connect dilation.
Parameters: src-is a input image matrix(uchar type)
output: results store in dst matrix(uchar type)
*/

int dilate(cv::Mat &src, cv::Mat &dst){
    for(int i=1; i<src.rows-1; i++){
        for(int j=1; j<src.cols-1; j++){
            if( (short)src.at<uchar>(i-1, j) < 5 || (short)src.at<uchar>(i, j-1) < 5 || (short)src.at<uchar>(i, j+1) 
            < 5 ||  (short)src.at<uchar>(i+1, j) < 5){
                dst.at<uchar>(i, j) = 0;
            }else{
                dst.at<uchar>(i, j) = 255;
            }
        }
    }

    return 0;
}

// Grassfire transform

int grassfireTransform(Mat& target, Mat& local, int c, int connectivity) {

	// pass one
	for (int i = 0; i < target.rows; ++i){
		for (int j = 0; j < target.cols; ++j){
			if (target.at<uchar>(i, j) == c){ 
                local.at<int>(i, j) = 0; 
                }
			else {
				int m = INT32_MAX;

				if (i > 0) { 
                    m = min(m, local.at<int>(i - 1, j) + 1); 
                }
				else { 
                    m = min(m, 1); 
                }
				if (j > 0) {
                    m = min(m, local.at<int>(i, j - 1) + 1); 
                }
				else { 
                    m = min(m,  1); 
                }

				if (connectivity == 8 && i > 0) {
					if (j > 0) { m = min(m, local.at<int>(i - 1, j - 1) + 1); 
                    }
					if (j < target.cols - 1) { m = min(m, local.at<int>(i - 1, j + 1) + 1); 
                    }
				}
				local.at<int>(i, j) = m;
			}
		}
	}

	// pass 2
	for (int i = target.rows - 1; i > -1; --i){
		for (int j = target.cols - 1; j > -1; --j){
			if (target.at<uchar>(i, j) == c){ 
                local.at<int>(i, j) = 0;
            }
			else {
				int m = local.at<int>(i, j);
				if (i < target.rows - 1){
                     m = min(m, local.at<int>(i + 1, j) + 1); 
                }
				else {
                     m = min(m,  1); 
                }
				if (j < target.cols - 1) { 
                    m = min(m, local.at<int>(i, j + 1) + 1);
                }
				else { 
                    m = min(m,  1); 
                }

				if (connectivity == 8 && i < target.rows - 1) {
					if (j < target.cols - 1) { 
                        m = min(m, local.at<int>(i + 1, j + 1) + 1); }
					if (j > 0) { 
                        m = min(m, local.at<int>(i + 1, j - 1) + 1); }
				}
				local.at<int>(i, j) = m;
			}
		}
	}
    return(0);
}

// Extn - Self implemented connected components code using union find

int findRoot(int ds[], int val){
    // union find function
    int parent = ds[val];
    while (parent != -1){
        val = parent;
        parent = ds[val];
    }
    return val;
}

std::vector<int> connectedComponentLabel(cv::Mat &src, cv::Mat &dst, cv::Mat &vis){
    // labels regions of a binary image, also returns the region IDs as a vector
    int regionID = 1;
    int unionFind[500];
    std::map<int, int> regionCounter;
    std::vector<int> regions;
    std::vector<cv::Scalar> regionColors = { cv::Scalar(228, 117, 83), cv::Scalar(83, 117, 228), cv::Scalar(30, 112, 255) };
    for(int k=0; k < 500 ; k++){    
        unionFind[k] = -1;
    }

    for(int i=1; i<src.rows-1; i++){
        for(int j=1; j<src.cols-1; j++){
            if( (short)src.at<uchar>(i, j) == 255 ){
                // fetch label of neighbor pixels
                int topLabel = (short)dst.at<uchar>(i-1, j);
                int leftLabel = (short)dst.at<uchar>(i, j-1);
                if(topLabel == 0 && leftLabel == 0){
                    dst.at<uchar>(i, j) = regionID;
                    regionID++;
                    regionCounter.insert(std::make_pair(regionID, 0));
                }else if(topLabel == 0){
                    dst.at<uchar>(i, j) = leftLabel;
                }else if(leftLabel == 0){
                    dst.at<uchar>(i, j) = topLabel;
                }else{
                    int minVal = std::min(topLabel, leftLabel);
                    int maxVal = std::max(topLabel, leftLabel);
                    dst.at<uchar>(i, j) = minVal;
                    if(minVal != maxVal){
                        unionFind[maxVal] = minVal;
                    }
                }
            }
        }
    }  

    for(int i=1; i<src.rows-1; i++){
        for(int j=1; j<src.cols-1; j++){
            if( dst.at<uchar>(i, j) != 0 && unionFind[dst.at<uchar>(i, j)] != -1 ){
                dst.at<uchar>(i, j) =  findRoot(unionFind, (short)dst.at<uchar>(i, j));
            }
        }
    }

    for(int i=1; i<src.rows-1; i++){
        for(int j=1; j<src.cols-1; j++){
            regionCounter[(short) dst.at<uchar>(i, j)] += 1;
        }
    }

    int colorIdx = 0;
    for(int m = 0; m < regionCounter.size(); m++){
        if( m != 0 && regionCounter[m] > 1000){
            std::cout<<m<<std::endl;
            // regionColors
            regions.push_back(m);
            for(int i=1; i<src.rows-1; i++){
                for(int j=1; j<src.cols-1; j++){
                    if( (short)dst.at<uchar>(i, j) == m){
                        vis.at<cv::Vec3b>(i, j)[0] = regionColors[colorIdx][0];
                        vis.at<cv::Vec3b>(i, j)[1] = regionColors[colorIdx][1];
                        vis.at<cv::Vec3b>(i, j)[2] = regionColors[colorIdx][2];
                    }
                }
            }
            colorIdx++;
        }
    }

    return regions;
}
 





