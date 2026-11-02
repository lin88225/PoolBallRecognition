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

const double edgeSmoothingSigma = 1.2;
const double edgeThresholdMedianMultiple = 5.0;
const double minimumEdgeThreshold = 4.0;
const double cannyLowThresholdFraction = 0.5;
const int voteRadiusCount = 5;
const double smallestVoteRadiusFraction = 0.88;
const double largestVoteRadiusFraction = 1.12;
const double voteSmoothingSigma = 2.0;

const double minimumOutlineScore = 0.25;
const double minimumNonClothCoverage = 0.70;
const double coverageDiscRadiusFraction = 0.70;
const double peakClearingRadiusFraction = 0.6;
const double minimumBallSeparation = 0.7 * ballDiameter;
const int maximumBallCount = 16;
const double cornerPocketExclusionRadius = 1.3 * ballDiameter;
const double middlePocketExclusionRadius = 36.0;
const double minimumCushionDistance = 0.5 * ballRadius;
const double colourDiscRadiusFraction = 0.80;
const double smallestWhiteSaturationLimit = 0.30 * 255;
const double whiteMinimumBrightnessRatio = 0.80;
const double darkMaximumBrightnessRatio = 0.40;
const double outerRingInnerRadiusFraction = 0.7;
const double clothRingInnerRadiusFraction = 1.3;
const double clothRingOuterRadiusFraction = 2.3;
const size_t minimumClothRingPixelCount = 30;

const double matchTolerance = ballRadius;

// Ground truth as row and column in table coordinates, with -1 for a ball that isn't on table.
// Each image lists balls 1 to 15 then the white ball. First image is AllBalls, then Table1 to Table6.
// Values are from supplied spreadsheet (second correction version)
const int groundTruth[7][ballCount][2] = {
  { {253,908},{250,959},{255,1013},{256,1072},{254,1132},{262,1184},{256,1244},{253,1307},
    {329,901},{330,961},{335,1013},{336,1067},{334,1130},{345,1187},{342,1245},{293,797} },
  { {309,1092},{429,1212},{383,1131},{335,1050},{285,1211},{261,1170},{352,1173},{331,1132},
    {240,1210},{310,1170},{407,1169},{281,1135},{374,1212},{360,1091},{331,1213},{288,29} },
  { {282,1103},{620,1224},{391,1121},{315,1046},{310,1305},{202,621},{361,1173},{331,1132},
    {177,725},{185,1269},{665,798},{296,438},{53,1235},{362,1090},{399,1205},{507,1169} },
  { {23,908},{616,1224},{447,1199},{336,29},{305,1406},{210,621},{260,1294},{226,1252},
    {177,725},{171,1270},{665,798},{249,351},{53,1233},{169,901},{399,1201},{434,29} },
  { {23,908},{546,1407},{444,1204},{-1,-1},{305,1406},{210,621},{260,1294},{226,1252},
    {181,724},{171,1270},{665,798},{249,351},{58,1231},{169,901},{399,1201},{88,1392} },
  { {535,23},{546,1407},{444,1204},{-1,-1},{305,1406},{131,25},{260,1294},{226,1252},
    {538,72},{171,1270},{665,798},{249,351},{-1,-1},{169,901},{399,1201},{608,26} },
  { {535,23},{477,1364},{444,1204},{-1,-1},{305,1406},{131,25},{260,1294},{226,1252},
    {538,72},{171,1270},{-1,-1},{249,351},{-1,-1},{169,901},{399,1201},{629,947} }
};
const char* imageFileNames[7] = { "AllBalls.jpeg", "Table1.jpeg", "Table2.jpeg", "Table3.jpeg",
                                  "Table4.jpeg", "Table5.jpeg", "Table6.jpeg" };
const char* imageNames[7] = { "AllBalls", "Table1", "Table2", "Table3", "Table4", "Table5", "Table6" };

// Name printed on a ball, where balls 0 to 14 are numbers 1 to 15 and ball 15 is white
string ballName(int ball) {
    return ball == 15 ? string("W") : to_string(ball + 1);
}

string formatNumber(double value, int decimals = 1) {
    char text[64];
    snprintf(text, sizeof(text), "%.*f", decimals, value);
    return string(text);
}

string formatPercentage(double fraction) {
    return formatNumber(100.0 * fraction, 1);
}

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
    Mat expectedRadius;

    Point2f toTable(Point2f point) const { return applyHomography(imageToTable, point); }
    Point2f toImage(Point2f point) const { return applyHomography(tableToImage, point); }

    float radiusAt(Point2f point) const {
        if (!(point.x == point.x) || !(point.y == point.y)) return expectedRadius.at<float>(0, 0);
        int column = min(max(cvRound(point.x), 0), expectedRadius.cols - 1);
        int row = min(max(cvRound(point.y), 0), expectedRadius.rows - 1);
        return expectedRadius.at<float>(row, column);
    }
};

// Finds perspective transformation from four corners. Expected ball radius at a pixel is
// Ball radius times image distance covered by one millimetre along table at that point
Geometry buildGeometry(Size imageSize) {
    Geometry geometry;
    vector<Point2f> imagePoints(imageCorners, imageCorners + 4), tablePoints(tableCorners, tableCorners + 4);
    geometry.imageToTable = getPerspectiveTransform(imagePoints, tablePoints);
    geometry.tableToImage = geometry.imageToTable.inv();
    geometry.expectedRadius.create(imageSize, CV_32F);
    vector<Point2f> imageRow(imageSize.width), tableRow, nextImageRow;
    for (int row = 0; row < imageSize.height; row++) {
        for (int column = 0; column < imageSize.width; column++) imageRow[column] = Point2f((float)column, (float)row);
        perspectiveTransform(imageRow, tableRow, geometry.imageToTable);
        for (Point2f& tablePoint : tableRow) tablePoint.x += 1.f;
        perspectiveTransform(tableRow, nextImageRow, geometry.tableToImage);
        float* radiusRow = geometry.expectedRadius.ptr<float>(row);
        for (int column = 0; column < imageSize.width; column++) {
            float radius = (float)(ballRadius * norm(nextImageRow[column] - imageRow[column]));
            if (!(radius > 3.f) || !(radius < 200.f)) radius = 3.f;
            radiusRow[column] = radius;
        }
    }
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

struct EdgeImages {
    Mat edges;
    Mat gradientX, gradientY;
    Mat magnitude;
    float threshold = 0;
};

// Smooths image and keeps gradient of colour channel that changes most at each pixel.
// Threshold is a multiple of median gradient and Canny thins edges by non-maxima suppression
EdgeImages computeEdges(const Mat& image, const Mat& regionMask) {
    EdgeImages edgeImages;
    Mat smoothed;
    GaussianBlur(image, smoothed, Size(0, 0), edgeSmoothingSigma);
    vector<Mat> channels;
    split(smoothed, channels);
    edgeImages.magnitude = Mat::zeros(image.size(), CV_32F);
    edgeImages.gradientX = Mat::zeros(image.size(), CV_32F);
    edgeImages.gradientY = Mat::zeros(image.size(), CV_32F);
    for (Mat& channel : channels) {
        Mat derivativeX, derivativeY, channelMagnitude;
        Sobel(channel, derivativeX, CV_32F, 1, 0, 3, 1.0 / 8);
        Sobel(channel, derivativeY, CV_32F, 0, 1, 3, 1.0 / 8);
        magnitude(derivativeX, derivativeY, channelMagnitude);
        Mat stronger = channelMagnitude > edgeImages.magnitude;
        channelMagnitude.copyTo(edgeImages.magnitude, stronger);
        derivativeX.copyTo(edgeImages.gradientX, stronger);
        derivativeY.copyTo(edgeImages.gradientY, stronger);
    }

    vector<float> magnitudes;
    for (int row = 0; row < image.rows; row += 3)
        for (int column = 0; column < image.cols; column += 3)
            if (regionMask.at<uchar>(row, column)) magnitudes.push_back(edgeImages.magnitude.at<float>(row, column));
    edgeImages.threshold = (float)max(edgeThresholdMedianMultiple * medianValue(magnitudes), minimumEdgeThreshold);

    Canny(smoothed, edgeImages.edges, 8 * cannyLowThresholdFraction * edgeImages.threshold, 8 * edgeImages.threshold, 3, true);
    edgeImages.edges &= regionMask;
    return edgeImages;
}

// Hough for circles of known radius. Every edge pixel votes for 2 points at expected ball
// radius along its gradient line, rather than for a whole circle of points, which keeps clutter low.
// votes are smoothed and divided by circumferenece so a complete circle scores about 1.
// Division is done in double precision so every pixel rounds exactly as a plain float division does
Mat circleScore(const EdgeImages& edgeImages, const Mat& expectedRadius) {
    Size imageSize = edgeImages.edges.size();
    Mat votes = Mat::zeros(imageSize, CV_32F);
    const float voteWeight = 1.f / voteRadiusCount;
    for (int row = 0; row < imageSize.height; row++)
        for (int column = 0; column < imageSize.width; column++) {
            if (!edgeImages.edges.at<uchar>(row, column)) continue;
            float magnitude = edgeImages.magnitude.at<float>(row, column);
            if (!(magnitude > 0)) continue;
            float directionX = edgeImages.gradientX.at<float>(row, column) / magnitude;
            float directionY = edgeImages.gradientY.at<float>(row, column) / magnitude;
            float radius = expectedRadius.at<float>(row, column);
            for (int side = -1; side <= 1; side += 2)
                for (int step = 0; step < voteRadiusCount; step++) {
                    float fraction = (float)(smallestVoteRadiusFraction + (largestVoteRadiusFraction - smallestVoteRadiusFraction) * step / (voteRadiusCount - 1));
                    int centreColumn = cvRound(column + side * fraction * radius * directionX);
                    int centreRow = cvRound(row + side * fraction * radius * directionY);
                    if (centreColumn >= 0 && centreRow >= 0 && centreColumn < imageSize.width && centreRow < imageSize.height)
                        votes.at<float>(centreRow, centreColumn) += voteWeight;
                }
        }
    Mat smoothedVotes;
    GaussianBlur(votes, smoothedVotes, Size(0, 0), voteSmoothingSigma);
    smoothedVotes *= (float)(2 * CV_PI * voteSmoothingSigma * voteSmoothingSigma);
    Mat radiusInDouble, circumferenceInDouble, circumference, roundedCircumference, votesInDouble, scoreInDouble, score;
    expectedRadius.convertTo(radiusInDouble, CV_64F);
    multiply(radiusInDouble, Scalar(2 * CV_PI), circumferenceInDouble);
    circumferenceInDouble.convertTo(circumference, CV_32F);
    circumference.convertTo(roundedCircumference, CV_64F);
    smoothedVotes.convertTo(votesInDouble, CV_64F);
    divide(votesInDouble, roundedCircumference, scoreInDouble);
    scoreInDouble.convertTo(score, CV_32F);
    patchNaNs(score, 0);
    return score;
}

// Fraction of a disc that lies on foreground. Disc is drawn in a small patch round its
// centre, cut to image, rather than in a blank image size of picture
double discCoverage(const Mat& foreground, Point2f centre, double discRadius) {
    Point discCentre(centre);
    int discRadiusPixels = cvRound(discRadius);
    Rect patch = Rect(discCentre.x - discRadiusPixels, discCentre.y - discRadiusPixels, 2 * discRadiusPixels + 1, 2 * discRadiusPixels + 1) &
                 Rect(Point(0, 0), foreground.size());
    if (patch.area() <= 0) return 0.0;
    Mat disc = Mat::zeros(patch.size(), CV_8U);
    circle(disc, discCentre - patch.tl(), discRadiusPixels, Scalar(255), FILLED);
    int pixelCount = countNonZero(disc);
    return pixelCount ? (double)countNonZero(foreground(patch) & disc) / pixelCount : 0.0;
}

// Whether a ball centre could really be at this table position. A ball can't overlap a cushion, so its
// centre is at least half a radius from it, allowing for centre seen from above being shifted a
// little away from camera and it can't be over a pocket mouth
bool isInsideTable(Point2f tablePoint) {
    if (tablePoint.x < minimumCushionDistance || tablePoint.y < minimumCushionDistance ||
        tablePoint.x > tableColumns - minimumCushionDistance || tablePoint.y > tableRows - minimumCushionDistance) return false;
    const Point2f cornerPockets[4] = { Point2f(0, 0), Point2f(0, tableRows), Point2f(tableColumns, 0), Point2f(tableColumns, tableRows) };
    for (const Point2f& pocket : cornerPockets)
        if (hypot(tablePoint.x - pocket.x, tablePoint.y - pocket.y) < cornerPocketExclusionRadius) return false;
    const Point2f middlePockets[2] = { Point2f(tableColumns / 2.f, 0), Point2f(tableColumns / 2.f, tableRows) };
    for (const Point2f& pocket : middlePockets)
        if (hypot(tablePoint.x - pocket.x, tablePoint.y - pocket.y) < middlePocketExclusionRadius) return false;
    return true;
}

struct BallFeatures {
    double outerRingWhiteFraction = 0;
    double darkFraction = 0;
    double hue = 0, saturation = 0, value = 0;
};

struct Detection {
    Point2f imagePosition, tablePosition;
    float radius = 0;
    double outlineScore = 0;
    BallFeatures features;
    int predictedBall = -1;
};

struct ImageAnalysis {
    string name;
    Mat image, hsvImage;
    Geometry geometry;
    ClothModel cloth;
    Mat clothDistanceImage, foreground, score;
    EdgeImages edges;
    vector<Detection> detections;
};

// Median brightness of the cloth in a ring around a ball. As in shadow detection, a ball's
// brightness is judged as a ratio to this, so changes in lighting across table cancel out
float localClothBrightness(const ImageAnalysis& analysis, Point2f centre, double radius) {
    vector<float> brightness;
    double ringInner = clothRingInnerRadiusFraction * radius, ringOuter = clothRingOuterRadiusFraction * radius;
    for (int row = max(0, (int)(centre.y - ringOuter)); row <= min(analysis.image.rows - 1, (int)(centre.y + ringOuter)); row++)
        for (int column = max(0, (int)(centre.x - ringOuter)); column <= min(analysis.image.cols - 1, (int)(centre.x + ringOuter)); column++) {
            double distanceFromCentre = hypot(column - centre.x, row - centre.y);
            if (distanceFromCentre < ringInner || distanceFromCentre > ringOuter || analysis.foreground.at<uchar>(row, column)) continue;
            brightness.push_back(analysis.hsvImage.at<Vec3b>(row, column)[2]);
        }
    return brightness.size() >= minimumClothRingPixelCount ? (float)medianValue(brightness) : analysis.cloth.brightness;
}

// Measures a ball's appearance. Each pixel that isn't cloth or covered by a ball nearer camera
// (lower in image) is dark, white or coloured, judged by its saturation and by its brightness as a
// ratio to local cloth
BallFeatures extractFeatures(const ImageAnalysis& analysis, Point2f centre, double radius, double whiteSaturationLimit) {
    BallFeatures features;
    float clothBrightness = localClothBrightness(analysis, centre, radius);
    double colourDiscRadius = colourDiscRadiusFraction * radius;
    int left = max(0, (int)floor(centre.x - radius)), right = min(analysis.image.cols - 1, (int)ceil(centre.x + radius));
    int top = max(0, (int)floor(centre.y - radius)), bottom = min(analysis.image.rows - 1, (int)ceil(centre.y + radius));
    int discCount = 0, darkCount = 0, colouredCount = 0, ringCount = 0, ringWhiteCount = 0;
    double hueCosineSum = 0, hueSineSum = 0, saturationSum = 0, brightnessSum = 0;
    for (int row = top; row <= bottom; row++)
        for (int column = left; column <= right; column++) {
            double squaredDistance = (column - centre.x) * (column - centre.x) + (row - centre.y) * (row - centre.y);
            if (squaredDistance > radius * radius) continue;
            if (!analysis.foreground.at<uchar>(row, column)) continue;
            Vec3b pixel = analysis.hsvImage.at<Vec3b>(row, column);
            double brightnessRatio = pixel[2] / clothBrightness;
            bool isDark = brightnessRatio <= darkMaximumBrightnessRatio;
            bool isWhite = !isDark && pixel[1] <= whiteSaturationLimit && brightnessRatio >= whiteMinimumBrightnessRatio;
            if (squaredDistance >= outerRingInnerRadiusFraction * outerRingInnerRadiusFraction * radius * radius) {
                ringCount++;
                if (isWhite) ringWhiteCount++;
            }
            if (squaredDistance > colourDiscRadius * colourDiscRadius) continue;
            discCount++;
            if (isDark) darkCount++;
            else if (!isWhite) {
                colouredCount++;
                hueCosineSum += cos(pixel[0] * CV_PI / 90.0); hueSineSum += sin(pixel[0] * CV_PI / 90.0);
                saturationSum += pixel[1]; brightnessSum += pixel[2];
            }
        }
    if (discCount > 0) features.darkFraction = (double)darkCount / discCount;
    if (ringCount > 0) features.outerRingWhiteFraction = (double)ringWhiteCount / ringCount;
    if (colouredCount > 0) {
        features.hue = fmod(atan2(hueSineSum, hueCosineSum) * 90.0 / CV_PI + 180.0, 180.0);
        features.saturation = saturationSum / colouredCount;
        features.value = brightnessSum / colouredCount;
    }
    return features;
}

struct Classifier {
    double cueThreshold = 0.8, blackThreshold = 0.6, stripeThreshold = 0.3;
    BallFeatures prototypes[15];
};

// Colour difference in hue, saturation and value, each in its natural units (degrees and percent of
// largest value), so no colour property is favoured. Hue is an angle, so difference wraps round
double colourDistance(const BallFeatures& first, const BallFeatures& second) {
    double hueGap = fabs(first.hue - second.hue) * 2.0;
    if (hueGap > 180) hueGap = 360 - hueGap;
    double saturationGap = (first.saturation - second.saturation) * 100.0 / 255.0;
    double valueGap = (first.value - second.value) * 100.0 / 255.0;
    return sqrt(hueGap * hueGap + saturationGap * saturationGap + valueGap * valueGap);
}

double meanOfFeature(const vector<BallFeatures>& features, int firstBall, int lastBall, double BallFeatures::*feature) {
    double sum = 0;
    for (int ball = firstBall; ball <= lastBall; ball++) sum += features[ball].*feature;
    return sum / (lastBall - firstBall + 1);
}

// Sets each threshold halfway between classes it separates. White of cue ball and of
// stripes reaches outline of ball, while number circle of a solid sits on its face, so
// outer ring white fraction is used for cue ball and for stripes
Classifier trainClassifier(const vector<BallFeatures>& features) {
    Classifier classifier;
    double mostWhiteOther = 0;
    for (int ball = 0; ball < 15; ball++) mostWhiteOther = max(mostWhiteOther, features[ball].outerRingWhiteFraction);
    classifier.cueThreshold = features[15].outerRingWhiteFraction > mostWhiteOther
        ? (features[15].outerRingWhiteFraction + mostWhiteOther) / 2 : 0.9 * features[15].outerRingWhiteFraction;
    double mostDarkOther = 0;
    for (int ball = 0; ball < 15; ball++) if (ball != 7) mostDarkOther = max(mostDarkOther, features[ball].darkFraction);
    classifier.blackThreshold = features[7].darkFraction > mostDarkOther
        ? (features[7].darkFraction + mostDarkOther) / 2 : 0.5 * features[7].darkFraction;
    double solidWhite = meanOfFeature(features, 0, 6, &BallFeatures::outerRingWhiteFraction);
    double stripeWhite = meanOfFeature(features, 8, 14, &BallFeatures::outerRingWhiteFraction);
    classifier.stripeThreshold = (solidWhite + stripeWhite) / 2;
    for (int ball = 0; ball < 15; ball++) classifier.prototypes[ball] = features[ball];
    return classifier;
}

// Names a ball. Solids are only compared with solids and stripes with stripes, because a stripe's band
// blends with white beside it and so looks a little different from solid of same colour
int classify(const Classifier& classifier, const BallFeatures& features) {
    if (features.outerRingWhiteFraction > classifier.cueThreshold) return 15;
    if (features.darkFraction > classifier.blackThreshold) return 7;
    int bestBall = 0;
    double bestDistance = 1e18;
    for (int ball = 0; ball < 15; ball++) {
        double distanceToPrototype = colourDistance(features, classifier.prototypes[ball]);
        if (distanceToPrototype < bestDistance) { bestDistance = distanceToPrototype; bestBall = ball; }
    }
    return bestBall;
}

// Stage 1 for one image
// Takes the strongest circle, clears round it and goes again
ImageAnalysis analyseImage(const Mat& image, const string& name) {
    ImageAnalysis analysis;
    analysis.name = name;
    analysis.image = image;
    analysis.geometry = buildGeometry(image.size());
    cvtColor(image, analysis.hsvImage, COLOR_BGR2HSV);

    Mat innerMask = tableMask(image.size(), innerClothGrowth);
    analysis.cloth = estimateCloth(analysis.hsvImage, innerMask);
    analysis.clothDistanceImage = clothDistance(analysis.hsvImage, analysis.cloth);
    analysis.foreground = (analysis.clothDistanceImage > 1.f);
    analysis.foreground.convertTo(analysis.foreground, CV_8U);
    morphologyEx(analysis.foreground, analysis.foreground, MORPH_OPEN, getStructuringElement(MORPH_ELLIPSE, Size(3, 3)));
    morphologyEx(analysis.foreground, analysis.foreground, MORPH_CLOSE, getStructuringElement(MORPH_ELLIPSE, Size(5, 5)));

    analysis.edges = computeEdges(image, tableMask(image.size(), edgeRegionGrowth));
    analysis.score = circleScore(analysis.edges, analysis.geometry.expectedRadius);

    Mat remainingScore = analysis.score.clone();
    remainingScore.setTo(0, tableMask(image.size(), searchAreaGrowth) == 0);
    for (int attempt = 0; attempt < 20 * maximumBallCount && (int)analysis.detections.size() < maximumBallCount; attempt++) {
        double peakScore;
        Point peakLocation;
        minMaxLoc(remainingScore, 0, &peakScore, 0, &peakLocation);
        cout << "peak " << peakScore << " at " << peakLocation << endl;
        if (peakScore < minimumOutlineScore) break;
        float radius = analysis.geometry.radiusAt(Point2f(peakLocation));
        Point2f centre(peakLocation);
        circle(remainingScore, peakLocation, cvRound(peakClearingRadiusFraction * radius), Scalar(0), FILLED);

        Point2f tablePoint = analysis.geometry.toTable(centre);
        if (!isInsideTable(tablePoint)) continue;
        bool tooClose = false;
        for (const Detection& other : analysis.detections)
            if (hypot(other.tablePosition.x - tablePoint.x, other.tablePosition.y - tablePoint.y) < minimumBallSeparation) tooClose = true;
        if (tooClose) continue;

        double coverage = discCoverage(analysis.foreground, centre, coverageDiscRadiusFraction * radius);
        if (coverage < minimumNonClothCoverage) continue;

        Detection detection;
        detection.imagePosition = centre;
        detection.tablePosition = tablePoint;
        detection.radius = radius;
        detection.outlineScore = peakScore;
        analysis.detections.push_back(detection);
    }
    return analysis;
}

// How one image compares with ground truth. Detection status is 0 when correctly named, 1 when
// named wrongly and 2 for a false detection
struct ImageEvaluation {
    string name;
    int groundTruthCount = 0, detectionCount = 0;
    int truePositives = 0, falsePositives = 0, falseNegatives = 0;
    vector<double> errors;
    int located = 0, correctNames = 0, correctTypes = 0;
    vector<int> detectionStatus;
    vector<int> detectionTruth;
    vector<int> missedBalls;
};

// Type of a ball, 0 for a solid, 1 for a stripe, 2 for black and 3 for white
int ballType(int ball) {
    return ball == 15 ? 3 : (ball == 7 ? 2 : (ball < 7 ? 0 : 1));
}

// Matches detections to ground truth one to one, closest pairs first, within one ball radius, then
// counts the true positives, false positives and misses and compares names given
ImageEvaluation evaluate(const ImageAnalysis& analysis, int imageIndex, int nameCounts[ballCount][ballCount]) {
    ImageEvaluation evaluation;
    evaluation.name = imageNames[imageIndex];
    evaluation.detectionCount = (int)analysis.detections.size();
    vector<Point2f> truePositions(ballCount);
    vector<bool> present(ballCount, false);
    for (int ball = 0; ball < ballCount; ball++)
        if (groundTruth[imageIndex][ball][0] >= 0) {
            present[ball] = true;
            truePositions[ball] = Point2f((float)groundTruth[imageIndex][ball][1], (float)groundTruth[imageIndex][ball][0]);
            evaluation.groundTruthCount++;
        }
    vector<tuple<double, int, int>> pairs;
    for (int detection = 0; detection < evaluation.detectionCount; detection++)
        for (int ball = 0; ball < ballCount; ball++)
            if (present[ball]) {
                double gap = hypot(analysis.detections[detection].tablePosition.x - truePositions[ball].x,
                                   analysis.detections[detection].tablePosition.y - truePositions[ball].y);
                if (gap <= matchTolerance) pairs.push_back(make_tuple(gap, detection, ball));
            }
    sort(pairs.begin(), pairs.end());
    vector<int> detectionMatch(evaluation.detectionCount, -1), ballMatch(ballCount, -1);
    for (auto& pairing : pairs) {
        int detection = get<1>(pairing), ball = get<2>(pairing);
        if (detectionMatch[detection] < 0 && ballMatch[ball] < 0) {
            detectionMatch[detection] = ball;
            ballMatch[ball] = detection;
            evaluation.errors.push_back(get<0>(pairing));
        }
    }
    evaluation.detectionStatus.assign(evaluation.detectionCount, 2);
    evaluation.detectionTruth.assign(evaluation.detectionCount, -1);
    for (int detection = 0; detection < evaluation.detectionCount; detection++) {
        if (detectionMatch[detection] < 0) { evaluation.falsePositives++; continue; }
        evaluation.truePositives++;
        int ball = detectionMatch[detection];
        evaluation.detectionTruth[detection] = ball;
        evaluation.located++;
        int predicted = analysis.detections[detection].predictedBall;
        if (predicted >= 0) {
            nameCounts[ball][predicted]++;
            if (predicted == ball) evaluation.correctNames++;
            if (ballType(predicted) == ballType(ball)) evaluation.correctTypes++;
            evaluation.detectionStatus[detection] = (predicted == ball) ? 0 : 1;
        } else evaluation.detectionStatus[detection] = 0;
    }
    for (int ball = 0; ball < ballCount; ball++)
        if (present[ball] && ballMatch[ball] < 0) { evaluation.falseNegatives++; evaluation.missedBalls.push_back(ball); }
    return evaluation;
}

}

void MyApplication() {
    Mat trainingImage = loadImage(imageFileNames[0]);
    if (trainingImage.empty()) { cout << "couldn't find " << imageFileNames[0] << endl; return; }
    ImageAnalysis training = analyseImage(trainingImage, imageNames[0]);

    // Features of the 16 balls in AllBalls, taken at the spreadsheet positions
    double whiteSaturationLimit = smallestWhiteSaturationLimit;
    vector<BallFeatures> trainingFeatures(ballCount);
    for (int ball = 0; ball < ballCount; ball++) {
        Point2f centre = training.geometry.toImage(Point2f((float)groundTruth[0][ball][1], (float)groundTruth[0][ball][0]));
        trainingFeatures[ball] = extractFeatures(training, centre, training.geometry.radiusAt(centre), whiteSaturationLimit);
    }
    Classifier classifier = trainClassifier(trainingFeatures);

    for (int image = 0; image < 7; image++) {
        Mat picture = loadImage(imageFileNames[image]);
        if (picture.empty()) { cout << "couldn't find " << imageFileNames[image] << endl; continue; }
        ImageAnalysis analysis = analyseImage(picture, imageNames[image]);
        for (Detection& detection : analysis.detections) {
            detection.features = extractFeatures(analysis, detection.imagePosition, detection.radius, whiteSaturationLimit);
            detection.predictedBall = classify(classifier, detection.features);
        }
        int nameCounts[ballCount][ballCount] = {};
        ImageEvaluation evaluation = evaluate(analysis, image, nameCounts);
        cout << imageNames[image] << "  true positives " << evaluation.truePositives << ", false positives " << evaluation.falsePositives
             << ", missed " << evaluation.falseNegatives << ", named correctly " << evaluation.correctNames << " of " << evaluation.located << endl;
    }
}
