#include <opencv2/opencv.hpp>
#include <iostream>
#include <math.h>
#include <opencv2\core\core.hpp>
#include <opencv2\highgui\highgui.hpp>
#include <opencv2\imgproc\imgproc.hpp>
#include <stdlib.h>
#include "skeletonization.hpp"
using namespace cv;
using namespace std;
int main()
{
    // Initialize camera
    cv::VideoCapture video_capture(1);
    if (!video_capture.isOpened())
    {
        std::cerr << "Error: Could not open video capture device." << std::endl;
        return -1;
    }

    while (true)
    {
        // Capture frame-by-frame
        cv::Mat frame;
        video_capture >> frame;
        waitKey(5);
        if (frame.empty())
        {
            std::cerr << "Error: Could not read frame." << std::endl;
            break;
        }
        imshow("orginal",frame);
    if (cv::waitKey(1) == 'q')
        {
            break;
        }
    }
    // // When everything is done, release the capture
    video_capture.release();
    cv::destroyAllWindows();

    return 0;
}