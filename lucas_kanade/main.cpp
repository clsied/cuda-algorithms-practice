#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/core/types.hpp>
using namespace std;

void goodFeaturesToTrack_gpu(cv::InputArray image, cv::OutputArray corners, int maxCorners, double qualityLevel,
    double minDistance, cv::InputArray mask = cv::noArray(), int blockSize = 3, bool useHarrisDetector = false, double k = 0.04);

void calcOpticalFlowPyrLK_gpu(cv::InputArray prevImg, cv::InputArray nextImg, cv::InputArray prevPts,
		cv::InputOutputArray nextPts, cv::OutputArray status, cv::OutputArray err,
		cv::Size winSize = cv::Size(21, 21), int maxLevel = 3, 
        cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30, 0.01),
		int flags = 0, double minEigThreshold = 1e-4);


int main() {

    cv::Mat img1_bgr = cv::imread("C:/Users/clsie/Downloads/MPI-Sintel-training_images/training/clean/bandage_2/frame_0001.png");
    cv::Mat img2_bgr = cv::imread("C:/Users/clsie/Downloads/MPI-Sintel-training_images/training/clean/bandage_2/frame_0041.png");

    if (img1_bgr.empty() || img2_bgr.empty()) {
        std::cerr << "Failed to load images!" << std::endl;
        return -1;
    }

    cv::Mat img1_gray, img2_gray;
    cv::cvtColor(img1_bgr, img1_gray, cv::COLOR_BGR2GRAY);
    cv::cvtColor(img2_bgr, img2_gray, cv::COLOR_BGR2GRAY);

    std::vector<cv::Point2f> prevPts;
    int maxCorners = 100;
    double qualityLevel = 0.3;
    double minDistance = 7;
    int blockSize = 7;
    bool useHarrisDetector = false;
    float k = 0.04;

    goodFeaturesToTrack_gpu(img1_gray, prevPts, maxCorners, qualityLevel, minDistance,
                            cv::Mat(), blockSize, useHarrisDetector, k);
    if (prevPts.empty()) {
        std::cerr << "No corners found!" << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> nextPts;
    std::vector<uchar> status;
    std::vector<float> err;
    cv::Size winSize(15, 15);
    int maxLevel = 2;
    cv::TermCriteria criteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 10, 0.03);

    calcOpticalFlowPyrLK_gpu(img1_gray, img2_gray, prevPts, nextPts, status, err,
                             winSize, maxLevel, criteria);
    cv::Mat output = img1_bgr.clone();
    for (size_t i = 0; i < prevPts.size(); ++i) {
        if (status[i]) {
            cv::circle(output, prevPts[i], 3, cv::Scalar(0, 0, 255), -1);
            cv::line(output, prevPts[i], nextPts[i], cv::Scalar(0, 255, 0), 2);
        }
    }

    cv::imshow("Optical Flow", output);
    cv::imwrite("optical_flow_output.png", output);
    cv::waitKey(0);
    cv::destroyAllWindows();

    return 0;
}
