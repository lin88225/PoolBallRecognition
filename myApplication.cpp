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
const Point2f tableCorners[4] = { Point2f(0, 0), Point2f(0, tableRows - 1), Point2f(tableColumns - 1, 0), Point2f(tableColumns - 1, tableRows - 1) };

const double toleranceInDeviations = 4.0;
const double minimumClothHueTolerance = 4.0;
const double minimumClothSaturationTolerance = 20.0;
const double darkObjectBrightnessFraction = 0.30;
const double veryDarkClothDistance = 10.0;
const int minimumClothPixelCount = 500;

// How table area is grown (positive) or shrank (negative) in pixels for each use
const int innerClothGrowth = -25;
const int searchAreaGrowth = 15;
const int edgeRegionGrowth = 30;

// Starting half widths for the cloth's hue and saturation spread and iterative estimate
const float hueStartHalfWidth = 15.f;
const float saturationStartHalfWidth = 40.f;
const double outlierCutOffInDeviations = 3.0;
const int meanAndDeviationIterations = 6;

const char* imageFileNames[7] = { "AllBalls.jpeg", "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
                                  "Table4.jpeg", "Table5.jpeg", "Table6.jpeg" };
const char* imageNames[7] = { "AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6" };

// Middle value of a list, which reorders list
double medianValue(vector<float>& values) {
    if (values.empty()) return 0;
    size_t middle = values.size() / 2;
    nth_element(values.begin(), values.begin() + middle, values.end());
    return values[middle];
}

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

// Table area, grown outwards (positive) or shrunk inwards (negative) by a number of pixels
Mat tableMask(Size imageSize, int growth) {
    Mat mask = Mat::zeros(imageSize, CV_8U);
    Point outline[4] = { Point(imageCorners[0]), Point(imageCorners[2]), Point(imageCorners[3]), Point(imageCorners[1]) };
    fillConvexPoly(mask, outline, 4, Scalar(255));
    if (growth > 0) dilate(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(2 * growth + 1, 2 * growth + 1)));
    if (growth < 0) erode(mask, mask, getStructuringElement(MORPH_ELLIPSE, Size(-2 * growth + 1, -2 * growth + 1)));
    return mask;
}

struct ClothModel {
    float hue = 0, saturation = 0;
    float hueTolerance = 4, saturationTolerance = 20;
    float brightness = 128;
};

// Signed difference between 2 hues, where hue runs round from 0 to 180
double hueDifference(double first, double second) {
    double difference = first - second;
    while (difference > 90) difference -= 180;
    while (difference < -90) difference += 180;
    return difference;
}

// Mean and standard deviation found iteratively. Values more than three standard deviations from
// mean are left out and estimate is repeated, so balls and other outliers don't widen spread
void iterativeMeanAndDeviation(const vector<float>& values, float startHalfWidth, double& mean, double& deviation) {
    mean = 0;
    deviation = startHalfWidth / outlierCutOffInDeviations;
    Mat valueColumn(values, false), gap, kept;
    for (int iteration = 0; iteration < meanAndDeviationIterations; iteration++) {
        absdiff(valueColumn, Scalar(mean), gap);
        kept = gap <= outlierCutOffInDeviations * deviation;
        if (countNonZero(kept) < 10) break;
        Scalar keptMean, keptDeviation;
        meanStdDev(valueColumn, keptMean, keptDeviation, kept);
        mean = keptMean[0];
        deviation = max(1e-3, keptDeviation[0]);
    }
}

// The cloth's hue is peak of a smoothed hue histogram of middle of table hue and
// saturation tolerances come from their spread. A shadow mainly lowers brightness and leaves hue and
// saturation nearly unchanged, so shadowed cloth still counts as cloth.
ClothModel estimateCloth(const Mat& hsvImage, const Mat& innerMask) {
    ClothModel cloth;
    vector<float> brightnessValues;
    for (int row = 0; row < hsvImage.rows; row += 2)
        for (int column = 0; column < hsvImage.cols; column += 2)
            if (innerMask.at<uchar>(row, column)) brightnessValues.push_back(hsvImage.at<Vec3b>(row, column)[2]);
    cloth.brightness = (float)medianValue(brightnessValues);

    vector<Mat> hsvChannels;
    split(hsvImage, hsvChannels);
    Mat histogramMask = innerMask & (hsvChannels[2] >= 0.5 * cloth.brightness);
    Mat histogram, wrapped, smoothed;
    int histogramSize = 180, hueChannel = 0;
    float hueRange[] = { 0, 180 };
    const float* ranges[] = { hueRange };
    calcHist(&hsvImage, 1, &hueChannel, histogramMask, histogram, 1, &histogramSize, ranges);
    copyMakeBorder(histogram, wrapped, 2, 2, 0, 0, BORDER_WRAP);
    blur(wrapped, smoothed, Size(1, 5));
    Point peak;
    double peakHeight;
    minMaxLoc(smoothed.rowRange(2, 182), 0, &peakHeight, 0, &peak);
    cloth.hue = (float)peak.y;

    vector<Vec3b> clothPixels;
    for (int row = 0; row < hsvImage.rows; row += 2)
        for (int column = 0; column < hsvImage.cols; column += 2)
            if (histogramMask.at<uchar>(row, column)) clothPixels.push_back(hsvImage.at<Vec3b>(row, column));

    vector<float> hueDifferences;
    for (const Vec3b& pixel : clothPixels) hueDifferences.push_back((float)hueDifference(pixel[0], cloth.hue));
    double hueMean, hueDeviation;
    iterativeMeanAndDeviation(hueDifferences, hueStartHalfWidth, hueMean, hueDeviation);
    cloth.hue = (float)fmod(cloth.hue + hueMean + 180.0, 180.0);
    cloth.hueTolerance = (float)max(minimumClothHueTolerance, toleranceInDeviations * hueDeviation);

    vector<float> saturations;
    for (const Vec3b& pixel : clothPixels)
        if (fabs(hueDifference(pixel[0], cloth.hue)) <= cloth.hueTolerance) saturations.push_back(pixel[1]);
    cloth.saturation = (float)medianValue(saturations);
    vector<float> saturationDifferences;
    for (float saturation : saturations) saturationDifferences.push_back(saturation - cloth.saturation);
    double saturationMean, saturationDeviation;
    iterativeMeanAndDeviation(saturationDifferences, saturationStartHalfWidth, saturationMean, saturationDeviation);
    cloth.saturation += (float)saturationMean;
    cloth.saturationTolerance = (float)max(minimumClothSaturationTolerance, toleranceInDeviations * saturationDeviation);
    return cloth;
}

// How different each pixel is from cloth, scaled so that 1 is edge of the cloth tolerance.
// Very dark pixels can't be cloth even in shadow, so they get a large value
Mat clothDistance(const Mat& hsvImage, const ClothModel& cloth) {
    vector<Mat> channels;
    split(hsvImage, channels);
    Mat hue, saturation, hueGap, saturationGap, distanceImage;
    channels[0].convertTo(hue, CV_32F);
    channels[1].convertTo(saturation, CV_32F);
    absdiff(hue, Scalar(cloth.hue), hueGap);
    Mat wrappedHueGap = 180.0 - hueGap;
    hueGap = cv::min(hueGap, wrappedHueGap);
    absdiff(saturation, Scalar(cloth.saturation), saturationGap);
    Mat scaledHueGap = hueGap / cloth.hueTolerance, scaledSaturationGap = saturationGap / cloth.saturationTolerance;
    distanceImage = cv::max(scaledHueGap, scaledSaturationGap);
    distanceImage.setTo(veryDarkClothDistance, channels[2] < darkObjectBrightnessFraction * cloth.brightness);
    return distanceImage;
}

}

void MyApplication() {
    utils::fs::createDirectories(outputFolder);
    Mat image = loadImage(imageFileNames[0]);
    if (image.empty()) { cout << "couldn't find " << imageFileNames[0] << endl; return; }
    Mat hsvImage;
    cvtColor(image, hsvImage, COLOR_BGR2HSV);
    ClothModel cloth = estimateCloth(hsvImage, tableMask(image.size(), innerClothGrowth));
    cout << "cloth hue " << cloth.hue << " saturation " << cloth.saturation << endl;
    Mat foreground = clothDistance(hsvImage, cloth) > 1.f;
    morphologyEx(foreground, foreground, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    morphologyEx(foreground, foreground, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));
    imwrite(outputFolder + "/foreground.png", foreground);
}
