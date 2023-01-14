/***********************************Project 3: Real time 2D object Detection**********************************
 Created by Samruddhi Raut.
  
    File Name: training.cpp
    This file access the functions in features.cpp file and store the calculated features in features_database.csv.
     
    Instructions to run the file:
    make
    ./Project_3
 
*/

/****************************************Task5:Collect training data******************************************/

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream>
#include <string>
#include <opencv2/core/saturate.hpp>
#include "sam.h"
#include <cstdio>
#include <cstring>
#include <cstdlib>
#include <cmath>
#include "csv_util.cpp"


using namespace cv;
using namespace std;

 Mat img;

int main(int argc, char *argv[]) {
        // cv::VideoCapture *capdev;

        // // // open the video device
        // capdev = new cv::VideoCapture(0);
        // if( !capdev->isOpened() ) {
        //         printf("Unable to open video device\n");
        //         return(-1);
        // }

        // // get some properties of the image
        // cv::Size refS( (int) capdev->get(cv::CAP_PROP_FRAME_WIDTH ),
        //                 (int) capdev->get(cv::CAP_PROP_FRAME_HEIGHT));
        // printf("Expected size: %d %d\n", refS.width, refS.height);
              
        char filename[255] = "feature_training.csv";
        char image_filename[255] = "key";

        while(true) {
            // *capdev >> img; // get a new frame from the camera, treat as a stream
    
            vector<float> object_features(0);
        
            // if( img.empty() ) {
            //     printf("Image is empty\n");
            //     // break;
            // }   
        

    //Reading the images from directory
            img = cv::imread("/home/samruddhi/CV/Project_3/img/101.jpeg");
            

    //processes the input images each from the video stream / image directory and return the clean output

            cv::Mat finalImg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::Mat finalImg2(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            cv::Mat visualRegions(img.rows, img.cols, CV_8UC3, cv::Scalar(0, 0, 0));
    //Threshold the input image to obtain the thresholded output
            cv::Mat thresholdImg(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            thresholdImage(img, thresholdImg);
            imshow("Before cleaning " , thresholdImg);

    /*Shrink the input image to get the eroded output( it erodes away the boundaries
    of foreground object (Always try to keep foreground in white))*/
            cv::Mat eroded(img.rows, img.cols, CV_8UC1, cv::Scalar(0));
            erode(thresholdImg, eroded);

    /* Grow the input image to get the dilated output .
            It is just opposite of erosion. Here, a pixel element is '1' if 
            atleast one pixel under the kernel is '1'. So it increases 
            the white region in the image or size of foreground object increases. 
            here we are doing noise removal, so the erosion is followed by dilation.
    */    
            dilate(eroded, finalImg);
    /*The function cv::morphologyEx can perform advanced morphological transformations 
    using an erosion and dilation as basic operations.It is the difference between 
    dilation and erosion of an image.The result will look like the outline of the object. */

            morphologyEx(finalImg, finalImg, MORPH_CLOSE, getStructuringElement(MORPH_RECT,Size(60,60)));
            imshow("after cleaning " , finalImg);

    /*computes the connected components labeled image and also produces 
    a statistics output for each label.label the regions using connected components*/

            cv::Mat labels, stats, centroids;
            int number_labels = connectedComponentsWithStats(finalImg, labels, stats, centroids , 8);
            
    /*Finds contours in a binary image.
    The function retrieves contours from the binary image using the algorithm. 
    The contours are a useful tool for shape analysis and object detection and recognition. */
            cv::RNG rng(12345);                
            vector<vector<Point>> contours;
            findContours( finalImg , contours, RETR_TREE, CHAIN_APPROX_SIMPLE, Point(0, 0) );
            vector<RotatedRect> minm_rectangle( contours.size() );
            
            int instance = 0;
            int flag = 0 , flag2 = 0;
            int a = 0 , b = 0;

            for( size_t i = 0; i < contours.size(); i++ )
            {
    /*The function calculates and returns 
    the minimum-area bounding rectangle (possibly rotated) for a specified point set.*/
                minm_rectangle[i] = minAreaRect( contours[i] );
                cv::Point2f vertices[4];
                minm_rectangle[i].points(vertices);

                if (contourArea(contours[i]) > 3000){ //The function computes a contour area.
                    int count = 0;
                    for( int i = 0; i < 4; i++ ){
                        cv::Point2f points = vertices[i];
                        if(points.x > 10 && points.y > 10 && points.x< 1270 && points.y < 710 ){
                            count += 1;}
                        else{
                            continue;}
                    }
                        if (count == 4 ){
                        instance = 1;
                        a = i;
                        if(contourArea(contours[i]) > 50000 && contourArea(contours[i]) < 700000){/*here we are checking if any bigger contour available in image*/
                            flag2 = 1;
                            b = i;
                        }                        
                    }
                }
            }
            double percentage_area = 0 ;  
            double aspect_ratio = 0;

            Mat drawing = Mat::zeros( finalImg.size(), CV_8UC3 );
            for( size_t i = 0; i< contours.size(); i++ ){  
                Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );   
                
                drawContours( drawing, contours, (int)i, color ); //this function draws the contours.            
            }  
            if (instance == 1){
                    
                if( flag2 == 1 ){
                    a = b;}
                Scalar color = Scalar( rng.uniform(0, 256), rng.uniform(0,256), rng.uniform(0,256) );
            
                Point2f rectangle_points[4];  /* gets the rotated rectangle*/
                
                minm_rectangle[a].points(rectangle_points);  // get the four corners of the rectangle
                for ( int j = 0; j < 4; j++ )
                {
                    line( img,rectangle_points[j],rectangle_points[(j+1)%4], color );// this fn draws each side of rect and loop it over 4 times to get a complete rect.
                }                       
            } 

            imshow("Image_Contours", drawing);

    //calculations for % area filled
                                   
            percentage_area = (contourArea(contours[a]) / ( minm_rectangle[a].size.width * minm_rectangle[a].size.height ) );

    // calculating the aspect_ratio (ht/wdth)

            if ( minm_rectangle[a].size.width > minm_rectangle[a].size.height )
            {
            swap(minm_rectangle[a].size.width, minm_rectangle[a].size.height);
            aspect_ratio = ( minm_rectangle[a].size.width / minm_rectangle[a].size.height) ;}
            else {
                aspect_ratio = ( minm_rectangle[a].size.height / minm_rectangle[a].size.width);}
            
            imshow("Video", img);

    //To make the images translation,rotation,scale and mirror invariants calculate the Hu moments
            vector<Moments> mom_ents(contours.size());
            
            mom_ents[a] = moments(contours[a]);//The function computes moments, up to the 3rd order, of a vector shape or a rasterized shape. 

            float c = mom_ents[a].m02 / (mom_ents[a].m00 * mom_ents[a].m00);
            float d = mom_ents[a].m20 / (mom_ents[a].m00 * mom_ents[a].m00);

            if ( c > d){
                d = mom_ents[a].m20 / (mom_ents[a].m00 * mom_ents[a].m00);
                c = mom_ents[a].m02 / (mom_ents[a].m00 * mom_ents[a].m00);}
            
            double alpha ; 

            vector<Point2f> centr_mass(contours.size());
                        
            centr_mass[a] = Point2f( mom_ents[a].m10/mom_ents[a].m00 , mom_ents[a].m01/mom_ents[a].m00 ); 

            alpha = 0.5 * atan2 ( 2 * mom_ents[a].mu11 , mom_ents[a].mu20 - mom_ents[a].mu02  );//to calculate the orientation of the central axis using central moments
            
            Point2f p1  = Point2f(float(200 * cos(alpha) + centr_mass[a].x ), float( 200 * sin(alpha) + centr_mass[a].y)) ;
            Point2f p2  = Point2f(float(centr_mass[a].x - 200 * cos(alpha)  ), float( centr_mass[a].y - 200 * sin(alpha)  )) ;
            
            line( img, p1, p2, Scalar(0, 0 , 255) );
            imshow("image_with_cenrl_xis" , img);
     /*The Hu Moments obtained in the previous step have a large range.
     log transform given below is use to bring them in the same range*/ 
            double hu_moments[7];
            HuMoments(mom_ents[a], hu_moments);
             // cout << "blah"<< endl;
            
            for(int i = 0; i < 7; i++) {
            hu_moments[i] = -1 * copysign(1.0, hu_moments[i]) * log10(abs(hu_moments[i]));
            object_features.push_back(hu_moments[i]); }
            
            object_features.push_back(aspect_ratio);
            object_features.push_back(percentage_area);
            object_features.push_back(c);
            object_features.push_back(d);
            waitKey(0);
             
            for(int z = 0; z < object_features.size(); z++){
                 cout << object_features[z] << endl;}
                
            append_image_data_csv(filename, image_filename,object_features, 0);// writing the .csv to the given file
            break;        
    }
}





