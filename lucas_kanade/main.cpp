#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>

using namespace std;

void goodFeaturesToTrack_gpu(cv::InputArray image, cv::OutputArray corners, int maxCorners, double qualityLevel,
    double minDistance, cv::InputArray mask = cv::noArray(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04);

void calcOpticalFlowPyrLK(cv::InputArray prevImg, cv::InputArray nextImg, cv::InputArray prevPts,
		cv::InputOutputArray nextPts, cv::OutputArray status, cv::OutputArray err,
		cv::Size winSize = cv::Size(21, 21), int maxLevel = 3, 
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
		int flags = 0, double minEigThreshold = 1e-4);

int main() {

    cv::Mat img_bgr = cv::imread("C:/Users/clsie/Downloads/monkaa__frames_cleanpass/frames_cleanpass/family_x2/left/0158.png", cv::IMREAD_COLOR);
    if (img_bgr.empty()) {
        cerr << "Failed to load image!" << endl;
        return -1;
    }

    // covert to grayscale image
    cv::Mat img_gray;
    cv::cvtColor(img_bgr, img_gray, cv::COLOR_BGR2GRAY);
    img_gray.convertTo(img_gray, CV_32F, 1.0 / 255.0);

    cv::Mat corners;
    int maxCorners = 100;
    double qualityLevel = 0.3;
    double minDistance = 7;
    int blockSize = 7;
    float k = 0.04;
    bool useHarrisDetector = false;

    goodFeaturesToTrack_gpu(img_gray, corners, maxCorners, qualityLevel, minDistance, cv::Mat(), blockSize, useHarrisDetector, k);

    // draw corners
    cv::Mat output;
    img_bgr.copyTo(output);
    for (int i = 0; i < corners.rows; ++i) {
        cv::Point2f pt = corners.at<cv::Point2f>(i, 0);
        cv::circle(output, pt, 3, cv::Scalar(0, 0, 255), -1);
    }

    cv::imshow("Corners", output);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
