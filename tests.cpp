// tests.cpp
// Tests for myApplication.cpp
// It includes myApplication.cpp so it can reach the functions in the unnamed namespace
// Run it from the project folder with run_tests.sh

#include "myApplication.cpp"

// ---------------------------------------------------------------------------
// Small test harness
// ---------------------------------------------------------------------------

static int checkCount = 0, failureCount = 0;

#define CHECK(condition) \
    do { checkCount++; if (!(condition)) { failureCount++; cout << "  FAIL line " << __LINE__ << ": " #condition << endl; } } while (0)
#define CHECK_NEAR(first, second, tolerance) CHECK(fabs((double)(first) - (double)(second)) <= (tolerance))

static void runTest(const char* name, void (*test)()) {
    int failuresBefore = failureCount;
    test();
    cout << (failureCount == failuresBefore ? "ok     " : "FAILED ") << name << endl;
}
// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

static void testFormatting() {
    CHECK(formatNumber(3.14159, 2) == "3.14");
    CHECK(formatNumber(2.0, 0) == "2");
    CHECK(formatPercentage(0.5) == "50.0");
    CHECK(formatPercentage(1.0) == "100.0");
}

static void testMedian() {
    vector<float> values = { 5, 1, 3 };
    CHECK_NEAR(medianValue(values), 3, 1e-9);
    vector<float> nothing;
    CHECK_NEAR(medianValue(nothing), 0, 1e-9);
}

static void testHueDifferenceWrapsRound() {
    CHECK_NEAR(hueDifference(170, 10), -20, 1e-9);
    CHECK_NEAR(hueDifference(10, 170), 20, 1e-9);
    CHECK_NEAR(hueDifference(60, 60), 0, 1e-9);
}

static void testBallNamesAndTypes() {
    CHECK(ballName(0) == "1");
    CHECK(ballName(14) == "15");
    CHECK(ballName(15) == "W");
    CHECK(ballType(3) == 0);
    CHECK(ballType(10) == 1);
    CHECK(ballType(7) == 2);
    CHECK(ballType(15) == 3);
}

// ---------------------------------------------------------------------------
// Geometry
// ---------------------------------------------------------------------------

static const Size pictureSize(1931, 509);

static void testCornersMapToTableCorners() {
    Geometry geometry = buildGeometry(pictureSize);
    for (int corner = 0; corner < 4; corner++) {
        Point2f tablePoint = geometry.toTable(imageCorners[corner]);
        CHECK_NEAR(tablePoint.x, tableCorners[corner].x, 0.5);
        CHECK_NEAR(tablePoint.y, tableCorners[corner].y, 0.5);
    }
}

static void testHomographyRoundTrip() {
    Geometry geometry = buildGeometry(pictureSize);
    Point2f point(900, 300);
    Point2f back = geometry.toImage(geometry.toTable(point));
    CHECK_NEAR(back.x, point.x, 0.1);
    CHECK_NEAR(back.y, point.y, 0.1);
}

static void testBallsLookBiggerNearTheCamera() {
    Geometry geometry = buildGeometry(pictureSize);
    float farRadius = geometry.radiusAt(Point2f(960, 60));
    float nearRadius = geometry.radiusAt(Point2f(960, 440));
    CHECK(nearRadius > farRadius);
    CHECK(farRadius > 5.f && nearRadius < 100.f);
}

static void testTableMask() {
    Mat mask = tableMask(pictureSize, 0);
    CHECK(mask.at<uchar>(250, 950) == 255);
    CHECK(mask.at<uchar>(500, 100) == 0);
    CHECK(countNonZero(tableMask(pictureSize, -25)) < countNonZero(mask));
    CHECK(countNonZero(tableMask(pictureSize, 15)) > countNonZero(mask));
}

// ---------------------------------------------------------------------------
// Ground truth
// ---------------------------------------------------------------------------

static void testGroundTruthIsInsideTheTable() {
    for (int image = 0; image < 7; image++)
        for (int ball = 0; ball < ballCount; ball++) {
            int row = groundTruth[image][ball][0], column = groundTruth[image][ball][1];
            if (row < 0) { CHECK(column < 0); continue; }
            if (row >= tableRows || column >= tableColumns)
                cout << "  " << imageNames[image] << " ball " << ballName(ball) << " is at " << row << "," << column << endl;
            CHECK(row >= 0 && row < tableRows);
            CHECK(column >= 0 && column < tableColumns);
        }
}

static void testTrainingImageHasAllSixteenBalls() {
    for (int ball = 0; ball < ballCount; ball++) CHECK(groundTruth[0][ball][0] >= 0);
}

// ---------------------------------------------------------------------------
// Cloth colour model
// ---------------------------------------------------------------------------

static Mat flatHsv(Size size, Vec3b value) {
    return Mat(size, CV_8UC3, Scalar(value[0], value[1], value[2]));
}

static void testClothModelFindsFlatColour() {
    Mat hsv = flatHsv(Size(120, 120), Vec3b(60, 150, 200));
    Mat inner(hsv.size(), CV_8U, Scalar(255));
    ClothModel cloth = estimateCloth(hsv, inner);
    CHECK_NEAR(cloth.hue, 60, 1.0);
    CHECK_NEAR(cloth.saturation, 150, 1.0);
    CHECK_NEAR(cloth.brightness, 200, 1.0);
    CHECK(cloth.hueTolerance >= minimumClothHueTolerance);
    CHECK(cloth.saturationTolerance >= minimumClothSaturationTolerance);
}

static ClothModel testCloth() {
    ClothModel cloth;
    cloth.hue = 60; cloth.saturation = 150;
    cloth.hueTolerance = 5; cloth.saturationTolerance = 20;
    cloth.brightness = 200;
    return cloth;
}

static void testClothDistance() {
    Mat hsv = flatHsv(Size(3, 1), Vec3b(60, 150, 200));
    hsv.at<Vec3b>(0, 1) = Vec3b(100, 150, 200);
    hsv.at<Vec3b>(0, 2) = Vec3b(60, 150, 20);
    Mat distance = clothDistance(hsv, testCloth());
    CHECK(distance.at<float>(0, 0) < 1.f);
    CHECK(distance.at<float>(0, 1) > 1.f);
    CHECK_NEAR(distance.at<float>(0, 2), veryDarkClothDistance, 1e-6);
}

static void testShadowStillCountsAsCloth() {
    Mat hsv = flatHsv(Size(1, 1), Vec3b(60, 150, 100));
    CHECK(clothDistance(hsv, testCloth()).at<float>(0, 0) < 1.f);
}

// ---------------------------------------------------------------------------
// Circle finding
// ---------------------------------------------------------------------------

static void testCircleScoreFindsADisc() {
    Mat image(160, 200, CV_8UC3, Scalar(40, 120, 40));
    circle(image, Point(100, 80), 20, Scalar(230, 230, 230), FILLED);
    Mat region(image.size(), CV_8U, Scalar(255));
    Mat expectedRadius(image.size(), CV_32F, Scalar(20));
    Mat score = circleScore(computeEdges(image, region), expectedRadius);
    double highest;
    Point where;
    minMaxLoc(score, 0, &highest, 0, &where);
    CHECK_NEAR(where.x, 100, 2);
    CHECK_NEAR(where.y, 80, 2);
    CHECK(highest > minimumOutlineScore);
}

static void testCircleScoreIgnoresEmptyCloth() {
    Mat image(160, 200, CV_8UC3, Scalar(40, 120, 40));
    Mat region(image.size(), CV_8U, Scalar(255));
    Mat expectedRadius(image.size(), CV_32F, Scalar(20));
    double highest;
    minMaxLoc(circleScore(computeEdges(image, region), expectedRadius), 0, &highest);
    CHECK(highest < 0.01);
}

static void testDiscCoverage() {
    Mat foreground = Mat::zeros(100, 100, CV_8U);
    circle(foreground, Point(50, 50), 10, Scalar(255), FILLED);
    CHECK(discCoverage(foreground, Point2f(50, 50), 8) > 0.99);
    CHECK(discCoverage(foreground, Point2f(10, 10), 8) < 0.01);
    CHECK_NEAR(discCoverage(foreground, Point2f(-50, -50), 8), 0, 1e-9);
}

static void testIsInsideTable() {
    CHECK(isInsideTable(Point2f(tableColumns / 2.f, tableRows / 2.f)));
    CHECK(!isInsideTable(Point2f(2, 300)));
    CHECK(!isInsideTable(Point2f(20, 20)));
    CHECK(!isInsideTable(Point2f(tableColumns / 2.f, 10)));
    CHECK(!isInsideTable(Point2f(-5, 100)));
}

// ---------------------------------------------------------------------------

int main() {
    runTest("formatting", testFormatting);
    runTest("median", testMedian);
    runTest("hue difference wraps round", testHueDifferenceWrapsRound);
    runTest("ball names and types", testBallNamesAndTypes);

    runTest("corners map to table corners", testCornersMapToTableCorners);
    runTest("homography round trip", testHomographyRoundTrip);
    runTest("balls look bigger near the camera", testBallsLookBiggerNearTheCamera);
    runTest("table mask", testTableMask);

    runTest("ground truth is inside the table", testGroundTruthIsInsideTheTable);
    runTest("training image has all sixteen balls", testTrainingImageHasAllSixteenBalls);

    runTest("cloth model finds a flat colour", testClothModelFindsFlatColour);
    runTest("cloth distance", testClothDistance);
    runTest("shadow still counts as cloth", testShadowStillCountsAsCloth);

    runTest("circle score finds a disc", testCircleScoreFindsADisc);
    runTest("circle score ignores empty cloth", testCircleScoreIgnoresEmptyCloth);
    runTest("disc coverage", testDiscCoverage);
    runTest("is inside table", testIsInsideTable);
    cout << "\n" << checkCount << " checks, " << failureCount << " failed" << endl;
    return failureCount == 0 ? 0 : 1;
}
