// myApplication.cpp
// Pool table assignment, finding and recognising the balls

#include <opencv2/opencv.hpp>

using namespace cv;

void MyApplication() {
    Mat image = imread("Images/AllBalls.jpeg");
    imshow("AllBalls", image);
    waitKey(0);
}
