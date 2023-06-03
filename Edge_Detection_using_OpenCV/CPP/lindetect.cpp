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
int Kernel_size = 15;
int low_threshold = 40;
int high_threshold = 120;
int rho = 1.0;
int Threshold = 20;
double theta = CV_PI/180;
int minLineLength = 3;
int maxLineGap = 1;

//多项式拟合
bool polynomial_curve_fit(std::vector<cv::Point>& key_point, int n, cv::Mat& A)
{
//Number of key points
int N = key_point.size();
//构造矩阵X
cv::Mat X = cv::Mat::zeros(n + 1, n + 1, CV_64FC1);
for (int i = 0; i < n + 1; i++)
{
for (int j = 0; j < n + 1; j++)
{
for (int k = 0; k < N; k++)
{
X.at<double>(i, j) = X.at<double>(i, j) +
std::pow(key_point[k].y, i + j);
}
}
}
//构造矩阵Y
cv::Mat Y = cv::Mat::zeros(n + 1, 1, CV_64FC1);
for (int i = 0; i < n + 1; i++)
{
for (int k = 0; k < N; k++)
{
Y.at<double>(i, 0) = Y.at<double>(i, 0) +
std::pow(key_point[k].x, i) * key_point[k].x;
}
}
A = cv::Mat::zeros(n + 1, 1, CV_64FC1);
//求解矩阵A
cv::solve(X, Y, A, cv::DECOMP_LU);
return true;
}

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
        //imshow("orginal",frame);
        //Mat frame = imread("robot.png");
        // Convert to grayscale
        cv::Mat gray,thre;
        cv::cvtColor(frame, gray, cv::COLOR_BGR2GRAY);
        Mat out;
        //获取自定义核
        Mat element = getStructuringElement(MORPH_RECT, Size(100, 100)); //第一个参数MORPH_RECT表示矩形的卷积核，当然还可以选择椭圆形的、交叉型的
        //膨胀操作
        dilate(gray, out, element);
        //二值
        threshold(gray, thre, 50, 255, THRESH_BINARY);
        //imshow("er",thre);
        // Blur image to reduce noise
        cv::Mat blurred;
        cv::GaussianBlur(thre, blurred, cv::Size(Kernel_size, Kernel_size), 0);
         // detect the contours on the binary image using cv2.CHAIN_APPROX_NONE
        vector<vector<Point>> contours;
        vector<Vec4i> hierarchy;
        findContours( thre, contours, hierarchy, RETR_TREE, CHAIN_APPROX_NONE);
       
        // Perform Canny edge detection
        cv::Mat edges;
        cv::Canny(thre, edges, low_threshold, high_threshold);

        // // Perform Hough lines probabilistic transform
        // std::vector<cv::Vec4i> lines;
        // cv::HoughLinesP(edges, lines, rho, theta, Threshold, minLineLength, maxLineGap);
        // for (size_t x = 0; x < lines.size(); x++)
        // {
        // Vec4i l = lines[x];
        // Point pt1(l[0], l[1]);
        // Point pt2(l[2], l[3]);
        // polylines(frame, vector<Point>{pt1, pt2}, true, Scalar(0, 255, 0), 1, LINE_AA);
        // }
        // cv::imshow("line detect test", frame);
        //检测圆形
        vector<Vec3f> circles;
        double dp = 2; //
        double minDist = 50;  //两个圆心之间的最小距离
        double  param1 = 100;  //Canny边缘检测的较大阈值
        double  param2 = 100;  //累加器阈值
        int min_radius = 40;  //圆形半径的最小值
        int max_radius = 60;  //圆形半径的最大值
        HoughCircles(gray, circles, HOUGH_GRADIENT, dp, minDist, param1, param2,
            min_radius, max_radius);    
        int H = frame.rows;
        int W = frame.cols;
        int m,n;
    //图像中标记出圆形
    Mat Mask1 = Mat::zeros(frame.size(), CV_8UC1);
    for (size_t i = 0; i < circles.size(); i++)
        {
            //读取圆心
        Point center(cvRound(circles[i][0]), cvRound(circles[i][1]));
            //读取半径
            int radius = cvRound(circles[i][2]);
            //绘制圆心
            circle(edges, center, 3, Scalar(0, 255, 0), -1, 8, 0);
            //绘制圆
            circle(frame, center, radius, Scalar(0, 0, 255), 3, 8, 0);
        
        circle(Mask1, center, radius, Scalar(255, 255, 255), -1); 
         }
         
        Mat imgAddMask1 = thre.clone();
        cv::add(Mask1,thre,imgAddMask1);
        //cv::imshow("multiply", imgAddMask1);
        //逆转二值图
        Mat thre_inv;
        bitwise_not(imgAddMask1,thre_inv);
        //连通域
        RNG rng(10086);
        Mat outr;
        int number = connectedComponents(thre_inv, outr, 8, CV_16U);  //统计图像中连通域的个数
        // vector<Vec3b> colors;
        // for (int i = 0; i < number; i++)
        // {
        // //使用均匀分布的随机数确定颜色
        //     Vec3b vec3 = Vec3b(rng.uniform(0,256),rng.uniform(0,256),rng.uniform(0,256));
        //     colors.push_back(vec3);
        // }

    //以不同颜色标记出不同的连通域
        Mat connect = Mat::zeros(thre.size(), frame.type());
        int w = connect.cols;
        int h = connect.rows;
        Vec3b color=Vec3b(rng.uniform(0,256),rng.uniform(0,256),rng.uniform(0,256));
        for (int row = 0; row < h; row++)
        {
            for (int col = 0; col < w; col++)
            {
                int label = outr.at<uint16_t>(row, col);
                if (label == 1)  //背景不改变
                {
                    connect.at<Vec3b>(row, col)=color;
                }
                
            }
        }
        //cv::imshow("liantong",connect);

        //骨干提取
         if (sum(connect) == Scalar(0)) {
            break;
        }
        Mat skel = skeletonization(connect);
       //imshow("Skeleton Image", skel);
        //曲线拟合
        //输入拟合点
        std::vector<cv::Point> points;
        int rowmin=h;
        int colmin=w;
        int colmax=0;
        for (int col = 0; col < w; col++)
        {
            for (int row = 0; row < h; row++)
            {
                if (skel.at<uchar>(row, col) == 255)  //白色
                {
                    points.push_back(cv::Point(row, col));
                    if (row<rowmin)
                    {
                        rowmin=row;
                    }
                    if(col<colmin)
                    {
                        colmin=col;
                    }
                    if(col>colmax)
                    {
                        colmax=col;
                    }
                }
                
            }
        }
        for (int i=1; i<10;++i)
        {
            int gap=points.size()/10;
            Point center(cvRound(points[gap*i].y), cvRound(points[gap*i].x));
            cv::circle(frame, center, 3, cv::Scalar(0, 255, 255), -1);
        }
        // cv::Mat A;
        // int Num=3;
        // polynomial_curve_fit(points, Num, A);
        // std::cout << "A = " << A << std::endl;
        // //计算曲线拟合的点
        // std::vector<cv::Point> points_fitted;
        // for (int i=0; i < 10; ++i)
        // {
        //     Point2d ipt;
        //     ipt.x = i*(colmax-colmin)/10+colmin;
        //     ipt.y = 0;
        //     for (int j = 0; j < Num + 1; ++j)
        //     {
        //     ipt.y +=A.at<double>(j, 0)*pow(ipt.x, j); // 计算拟合函数的值
        //     }
        //     points_fitted.push_back(ipt);
        //     Point center(cvRound(points_fitted[i].y), cvRound(points_fitted[i].x));
        //     cv::circle(frame, center, 5, cv::Scalar(0, 255, 255), -1);
        // }
        //显示结果
       imshow("cricle", frame);

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
