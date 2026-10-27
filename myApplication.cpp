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

const int tableRows = 683;
const int tableColumns = 1434;
const double ballRadius = 21.5;
const double ballDiameter = 43.0;
const int ballCount = 16;

const bool showWindows = true;
const string outputFolder = "output";

const Point2f imageCorners[4] = { Point2f(337, 35), Point2f(38, 472), Point2f(1570, 21), Point2f(1888, 451) };
const Point2f tableCorners[4] = { Point2f(0, 0), Point2f(tableColumns - 1, 0), Point2f(0, tableRows - 1), Point2f(tableColumns - 1, tableRows - 1) };

const char* imageFileNames[7] = { "AllBalls.jpeg", "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
                                  "Table4.jpeg", "Table5.jpeg", "Table6.jpeg" };
const char* imageNames[7] = { "AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6" };

Point2f applyHomography(const Mat& homography, Point2f point) {
    vector<Point2f> points = { point }, transformed;
    perspectiveTransform(points, transformed, homography);
    return transformed[0];
}

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

// Mapping between camera image and table and radius a ball is expected to have at
// every image position
struct Geometry {
    Mat imageToTable, tableToImage;

    Point2f toTable(Point2f point) const { return applyHomography(imageToTable, point); }
    Point2f toImage(Point2f point) const { return applyHomography(tableToImage, point); }
};

// Finds perspective transformation from four corners
Geometry buildGeometry(Size imageSize) {
    Geometry geometry;
    vector<Point2f> imagePoints(imageCorners, imageCorners + 4), tablePoints(tableCorners, tableCorners + 4);
    geometry.imageToTable = getPerspectiveTransform(imagePoints, tablePoints);
    geometry.tableToImage = geometry.imageToTable.inv();
    return geometry;
}

}

void MyApplication() {
    utils::fs::createDirectories(outputFolder);
    Mat image = loadImage(imageFileNames[0]);
    if (image.empty()) { cout << "couldn't find " << imageFileNames[0] << endl; return; }
    Geometry geometry = buildGeometry(image.size());
    Mat flattened;
    warpPerspective(image, flattened, geometry.imageToTable, Size(tableColumns, tableRows));
    imwrite(outputFolder + "/flattened.png", flattened);
    imshow("flattened", flattened);
    waitKey(0);
}
