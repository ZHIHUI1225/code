
#ifndef LIB_VISULIZE_H
#define LIB_VISULIZE_H

#include <ros/ros.h>
#include <image_transport/image_transport.h>
#include "nav_msgs/Odometry.h"
#include  "sensor_msgs/Imu.h"
#include <time.h>
#include <stdlib.h>
#include "Eigen/Dense"
#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"
#include <ros/callback_queue.h>
#include <boost/thread/thread.hpp>
#include <geometry_msgs/Point.h>
#include <mutex>
#include <ros/ros.h>
#include <tf/tf.h>
#include <iostream>
#include <fstream>
#include <stdexcept>
#include <cfloat>
#include <fstream>
#include <ctime>
#include <string>
#include <bits/stdc++.h>

#include <ros/ros.h>
#include <vector>
#include <iostream>
#include <fstream>
#include <list>
#include <math.h>
#include <assert.h>

#include <tf/tf.h>
#include <tf/transform_listener.h>
#include <sensor_msgs/LaserScan.h>
#include <nav_msgs/OccupancyGrid.h>
#include <visualization_msgs/MarkerArray.h>

#include <sensor_msgs/Image.h>
#include "opencv2/opencv.hpp"
#include "cv_bridge/cv_bridge.h"

#include <sensor_msgs/PointCloud.h>
#include <sensor_msgs/PointCloud2.h>
#include <sensor_msgs/point_cloud_conversion.h>

#include <cv_bridge/cv_bridge.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <Eigen/Eigen>
#include <opencv2/aruco.hpp>
#include <tf/transform_listener.h>
#include <robot_msg/robot_pose.h>
#include <robot_msg/robot_pose_array.h>
#include <robot_msg/target_pose.h>
#include <robot_msg/target_pose_array.h>


using namespace std;

class Visualize //速度控制类
{
    public:
    Visualize();
    ~Visualize();
    /** Functions **/
    /**
     * @brief 管理unitree_nav_node节点的运行
     * 程序运行
     */
    void Execute();

    private:
    /** Const **/
    const int ROS_RATE_HZ = 50; //ROS更新频率

    /** Node Handle **/
    ros::NodeHandle n; // 节点句柄

	ros::Subscriber line_sub, image_sub;
	ros::Publisher line_pub, poly_pub, line_without_QR_pub, line_in_QR_pub,tube_pub,  position_pub, QR_1_pub_, QR_2_pub_;
	ros::Publisher tf_pc_pub_, robot_pub_;
	cv::Mat img_left_;
	cv::Mat img_right_;
	sensor_msgs::PointCloud poly_pc_;
	robot_msg::robot_pose robot_;

    /** Parameters **/

    /** Variables **/
	/** Functions **/
	void kalmanFilter();
		inline void initializeQuaternion(geometry_msgs::Quaternion& quaternion)
		{
			quaternion.x = 0;
			quaternion.y = 0;
			quaternion.z = 0;
			quaternion.w = 1;
		}
		bool transformPosePositioin(geometry_msgs::Pose& input,
									geometry_msgs::Pose& output,
									tf::TransformListener* listener);
		boost::recursive_mutex mutex_;
		tf::TransformListener* listener_; // TF接收
		void lineCallBack(const sensor_msgs::PointCloud2::ConstPtr& Input)
		{
			sensor_msgs::PointCloud line_pc;
			sensor_msgs::convertPointCloud2ToPointCloud(*Input, line_pc);
		}

		Eigen::VectorXf FitterLeastSquareMethod(vector<float> &X, vector<float> &Y, uint8_t orders);

		inline double getDigitTwo(double x)
		{
			int t = x * 100;
			return t * 0.01;
		}
		sensor_msgs::PointCloud pc_without_QR_;
		geometry_msgs::Pose QR_1_p, QR_2_p;
		void imageCallBack(const sensor_msgs::Image::ConstPtr& Input)
		{
			pc_without_QR_.points.clear();
			cv_bridge::CvImagePtr cv_ptr;
			cv_ptr = cv_bridge::toCvCopy(Input, sensor_msgs::image_encodings::BGR8);
			//680 480
			cv::Mat crop = cv_ptr->image(cv::Range(100, 360), cv::Range(100, 640));
			//cv::Mat crop = cv_ptr->image(cv::Range::all(), cv::Range::all());
			cv::imshow("image", crop);
			cv::waitKey(3);

			cv::Mat HSVMat, mask, mash_th, mash_dst;
			cv::cvtColor(crop, HSVMat, cv::COLOR_BGR2HSV);
			cv::Scalar scalarL = cv::Scalar(0, 0, 0); // black
			cv::Scalar scalarH = cv::Scalar(150, 200, 75); // black
			cv::inRange(crop, scalarL, scalarH, mask);

			cv::threshold(mask, mash_th, 0, 255, cv::THRESH_BINARY);
			cv::Mat kernel = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(6, 6));
			cv::dilate(mash_th, mash_dst, kernel);
			cv::imshow("image2", mash_dst);
			cv::waitKey(3);

			std::vector<std::vector<cv::Point>> contours;
			std::vector<cv::Point> contour;
			cv::findContours(mash_th, contours, cv::RETR_TREE, cv::CHAIN_APPROX_SIMPLE);
			sensor_msgs::PointCloud pc;
			pc.header.frame_id = "map";
			pc.header.stamp = ros::Time::now();
			pc_without_QR_.header.frame_id = "map";
			pc_without_QR_.header.stamp = ros::Time::now();
			std::vector<float> x, y;

			double i = 0;
			for (auto c : contours)
				for (auto cc : c)
				{
					i += 0.01;
					geometry_msgs::Point32 p;
					p.x = cc.x * 0.01;
					p.y = cc.y * 0.01;
					p.z = i;
					pc.points.push_back(p);
				}
			
			std::vector<int> markerIds;
			std::vector<std::vector<cv::Point2f>> markerCorners;
			cv::Ptr<cv::aruco::Dictionary> dictionary = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_6X6_250);

			cv::Mat cameraMatrix = (cv::Mat_<float>(3, 3) <<
				1.18346606e+03, 0.0, 3.14407234e+02,
				0.0, 1.18757422e+03, 2.38823696e+02,
				0.0, 0.0, 1);

			double fx, fy, cx, cy, k1, k2, k3, p1, p2;
			fx = 1.18346606e+03;
			fy = 1.18757422e+03;
			cx = 3.14407234e+02;
			cy = 2.38823696e+02;
			k1 = -0.51328742;
			k2 = 0.33232725;
			p1 = 0.01683581;
			p2 = -0.00078608;
			k3 = -0.1159959;
			cv::Mat distCoeffs = (cv::Mat_<float>(5, 1) << k1, k2, p1, p2, k3);
			std::vector<cv::Vec3d> rvecs, tvecs;
			
			cv::aruco::detectMarkers(crop, dictionary, markerCorners, markerIds);
			cv::aruco::estimatePoseSingleMarkers(markerCorners, 0.05, cameraMatrix, distCoeffs, rvecs, tvecs);

			//if (markerIds.size() < 2) return;
			cout << "markerIds.size() " << markerIds.size() << endl;

			robot_msg::robot_pose_array robot_array;

			if (markerIds.size() > 0) {
				cv::aruco::drawDetectedMarkers(crop, markerCorners, markerIds);
				for (int i = 0; i < rvecs.size(); ++i) {
					cv::aruco::drawAxis(crop, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
					robot_msg::robot_pose robot;
					robot.ID.data = markerIds[i];
					robot.position.x = (markerCorners[i][0].x + markerCorners[i][1].x + markerCorners[i][2].x + markerCorners[i][3].x)/4;
					robot.position.y = (markerCorners[i][0].y + markerCorners[i][1].y + markerCorners[i][2].y + markerCorners[i][3].y)/4;
					robot.yaw = atan2(markerCorners[i][0].y - markerCorners[i][3].y, markerCorners[i][0].x - markerCorners[i][3].x); // atan2(markerCorners[i][1].y - markerCorners[i][0].y, markerCorners[i][1].x - markerCorners[i][0].x)
					cout << i << " ID:" << markerIds[i] << endl;
					cout << " (" << robot.position.x << ", " << robot.position.y << ") " << robot.yaw << endl; // robot.position.y 
					robot_array.robot_pose_array.push_back(robot);
					// cout << "i " << i << " " << rvecs[i] << " " << tvecs[i] << endl;
				}
			}
			robot_pub_.publish(robot_array);
			cv::imshow("out", crop);
			
			// for (int i = 0; i < markerIds.size(); ++i)
			// {

			// }

			
			QR_1_p.position.x = (markerCorners[0][0].x + markerCorners[0][1].x + markerCorners[0][2].x + markerCorners[0][3].x) / 400;
			QR_1_p.position.y = (markerCorners[0][0].y + markerCorners[0][1].y + markerCorners[0][2].y + markerCorners[0][3].y) / 400;
			QR_1_p.position.z = markerIds[0];
			QR_2_p.position.x = (markerCorners[1][0].x + markerCorners[1][1].x + markerCorners[1][2].x + markerCorners[1][3].x) / 400;
			QR_2_p.position.y = (markerCorners[1][0].y + markerCorners[1][1].y + markerCorners[1][2].y + markerCorners[1][3].y) / 400;
			QR_2_p.position.z = markerIds[1];

			for (int i = 0; i < markerIds.size(); ++i)
			{
				if (markerIds[i] == 13)
				{
					// cout << "find 13" << endl;
					sensor_msgs::PointCloud tf_pc;
					tf_pc.header.frame_id = "map";
					tf_pc.header.stamp = ros::Time::now();
					for (int j = 0; j < 4; ++j)
					{
						geometry_msgs::Point32 p;
						p.x = markerCorners[i][j].x/100;
						p.y = markerCorners[i][j].y/100;
						tf_pc.points.push_back(p);
					}
					// cout << "tf_pc " << tf_pc.points.size() << endl;
					tf_pc_pub_.publish(tf_pc);
				}
			}
			
			if (markerIds.size() > 0) {
				cv::aruco::drawDetectedMarkers(crop, markerCorners, markerIds);
				for (int i = 0; i < rvecs.size(); ++i) {
					cv::aruco::drawAxis(crop, cameraMatrix, distCoeffs, rvecs[i], tvecs[i], 0.1);
					//cout << "i " << i << " " << rvecs[i] << " " << tvecs[i] << endl;
				}
			}
			// geometry_msgs::Point32 qr1;
			// qr1.x = QR_1_p.position.x;
			// qr1.y = QR_1_p.position.y;
			// pc_without_QR_.points.push_back(qr1);

			pc_without_QR_.points.push_back(pc.points[i]);
			for (int i = 0; i < pc.points.size(); ++i)
			{
				if (hypot(pc.points[i].x - QR_1_p.position.x, pc.points[i].y - QR_1_p.position.y) > 0.27 && 
					hypot(pc.points[i].x - QR_2_p.position.x, pc.points[i].y - QR_2_p.position.y) > 0.27)
				{
					pc_without_QR_.points.push_back(pc.points[i]);
					//cout<<pc.points[i]<<" "<< pc.points.size() <<endl;
				}	

			}

			geometry_msgs::Point32 qr2;
			qr2.x = QR_2_p.position.x;
			qr2.y = QR_2_p.position.y;
			pc_without_QR_.points.push_back(qr2);

			QR_1_p.orientation.w = atan2(markerCorners[0][1].y - markerCorners[0][0].y, markerCorners[0][1].x - markerCorners[0][0].x);
			QR_2_p.orientation.w = atan2(markerCorners[1][1].y - markerCorners[1][0].y, markerCorners[1][1].x - markerCorners[1][0].x);
			QR_1_pub_.publish(QR_1_p);
			QR_2_pub_.publish(QR_2_p);

			line_without_QR_pub.publish(pc_without_QR_);
		
			cv::imshow("out", crop);

			line_pub.publish(pc);
		
		}
}; // Visualize

#endif // LIB_VISULIZE_H

