// myApplication.cpp
// Pool table assignment, finding and recognising the balls

#include <opencv2/opencv.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <string>
#include <tuple>
#include <vector>

using namespace cv;
using namespace std;

void MyApplication();

namespace {

const char* imageFileNames[7] = { "AllBalls.jpeg", "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
                                  "Table4.jpeg", "Table5.jpeg", "Table6.jpeg" };
const char* imageNames[7] = { "AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6" };

// Looks in current folder and usual image folders
Mat loadImage(const string& fileName) {
    static const char* folders[] = { "", "Images/", "images/", "../Images/", "../images/", "../",
                                     "../../Images/", "Resources/", "../Resources/" };
    for (const char* folder : folders) {
        string path = string(folder) + fileName;
        if (utils::fs::exists(path)) {
            Mat image = imread(path, IMREAD_COLOR);
            if (!image.empty()) return image;
        }
    }
    return Mat();
}

}

void MyApplication() {
    for (int image = 0; image < 7; image++) {
        Mat picture = loadImage(imageFileNames[image]);
        if (picture.empty()) { cout << "couldn't find " << imageFileNames[image] << endl; continue; }
        cout << imageFileNames[image] << " " << picture.cols << "x" << picture.rows << endl;
    }
}
