#ifndef _GO_CPP_
#define _GO_CPP_

#include "go.hpp"

using namespace std;
using namespace cv;

Visualize::Visualize()
{
	line_sub = n.subscribe("line",10, &Visualize::lineCallBack,this);
	image_sub = n.subscribe("/usb_cam/image_raw",10, &Visualize::imageCallBack,this);
  line_pub = n.advertise<sensor_msgs::PointCloud>("/line", 1);
  line_without_QR_pub = n.advertise<sensor_msgs::PointCloud>("/line_without_QR", 1);
  line_in_QR_pub = n.advertise<sensor_msgs::PointCloud>("/line_in_QR", 1);
  position_pub = n.advertise<geometry_msgs::Pose>("/robot_pose", 1);
  QR_1_pub_ = n.advertise<geometry_msgs::Pose>("/QR_1", 1);
  QR_2_pub_ = n.advertise<geometry_msgs::Pose>("/QR_2", 1);
  tf_pc_pub_ = n.advertise<sensor_msgs::PointCloud>("/tf_pc", 1);
  poly_pub = n.advertise<sensor_msgs::PointCloud>("/poly", 1);
  tube_pub = n.advertise<sensor_msgs::PointCloud>("/tube", 1);
  listener_ = new tf::TransformListener(n, ros::Duration(5.0), true);
  robot_pub_ = n.advertise<robot_msg::robot_pose_array>("/robot", 1);
}  

Visualize::~Visualize()
{
	cv::destroyAllWindows();
  ROS_INFO("VISULIZE NODE CLOSED");
}

void Visualize::Execute()
{
  ros::Rate loop_rate(10);

  while (ros::ok())
  {
    /* //zhushi
    geometry_msgs::Pose input;
    sensor_msgs::PointCloud pc_in_QR_;
    pc_in_QR_.header.frame_id = "QR";
    pc_in_QR_.header.stamp = ros::Time::now();
    double large_y = 0;
    double middle_x;
    for (int i = 0; i < pc_without_QR_.points.size(); ++i)
    {
      input.position.x = pc_without_QR_.points[i].x;
      input.position.y = pc_without_QR_.points[i].y;
      geometry_msgs::Pose output;
      transformPosePositioin(input, output, listener_);
      geometry_msgs::Point32 p;
      p.x = output.position.x;
      p.y = output.position.y;
      if (fabs(large_y) < fabs(p.y)) {large_y = p.y; middle_x = p.x;}
      pc_in_QR_.points.push_back(p);
    }

    line_in_QR_pub.publish(pc_in_QR_);

    vector<geometry_msgs::Point32> pc_v;

    for (int i = 0; i < pc_in_QR_.points.size(); ++i)
    {
      pc_v.push_back(pc_in_QR_.points[i]);
    }

    std::sort(pc_v.begin(), pc_v.end(), [](const geometry_msgs::Point32& a, const geometry_msgs::Point32& b) {
		  return a.x > b.x;//    <升序>降序
		});

    double length = 0.3;
    int k = 0;
    sensor_msgs::PointCloud pc_10;
    sensor_msgs::PointCloud pc_expand;
    for (int i = 1; i < pc_v.size(); ++i)
    pc_expand.points.push_back(pc_v[i]);

    for (int i = 1; i < pc_expand.points.size(); ++i)
    {
      double l = hypot(pc_v[k].x - pc_v[i].x, pc_v[k].y - pc_v[i].y);
      if (l > 0.1) { 
        pc_10.points.push_back(pc_v[k]);
        k = i;
      }
    }

    geometry_msgs::Point32 qr1;
    qr1.x = 0;
    qr1.y = 0;
    pc_10.points.push_back(qr1);

    pc_expand.header.frame_id = "map";
    pc_expand.header.stamp = ros::Time::now();
    pc_10.header.frame_id = "map";
    pc_10.header.stamp = ros::Time::now();
    if (pc_v.size() > 0)
    poly_pub.publish(pc_10);

    sensor_msgs::PointCloud pc_expanded;
    pc_expanded.header.frame_id = "map";
    pc_expanded.header.stamp = ros::Time::now();
    double tt = 0;
    for (int i = 1; i < pc_10.points.size(); ++i)
    {
      for (int m = 1; m < 50; ++m)
      {
        geometry_msgs::Point32 p;
        p.x = (pc_10.points[i].x - pc_10.points[i-1].x)*0.02*m + pc_10.points[i-1].x;
        p.y = (pc_10.points[i].y - pc_10.points[i-1].y)*0.02*m + pc_10.points[i-1].y;
        tt += 0.01;
        p.z = tt;
        pc_expanded.points.push_back(p);
      }
    }
    sensor_msgs::PointCloud pc_ave;
    geometry_msgs::Point32 ori;
    ori.x = 0;
    ori.y = 0;
    pc_ave.points.push_back(ori);
    pc_ave.header.frame_id = "QR";
    pc_ave.header.stamp = ros::Time::now();
    k = 0;
    double ll = 0;
    for (int i = 1; i < pc_10.points.size(); ++i)
    {
      ll += hypot(pc_10.points[i].x - pc_10.points[i-1].x,
                  pc_10.points[i].y - pc_10.points[i-1].y);
    }
    for (int i = 1; i < pc_expanded.points.size(); ++i)
    {
      double l = hypot(pc_expanded.points[k].x - pc_expanded.points[i].x, pc_expanded.points[k].y - pc_expanded.points[i].y);
      
      if (l > 0.16) { 
        pc_ave.points.push_back(pc_expanded.points[k]);
        k = i;
      }
    }

    if (pc_expanded.points.size() != 0)
    poly_pub.publish(pc_ave);
//zhushi
*/
    loop_rate.sleep();
    ros::spinOnce();
  }
}

bool Visualize::transformPosePositioin(geometry_msgs::Pose& input,
                                       geometry_msgs::Pose& output,
                                       tf::TransformListener* listener)
{	
  initializeQuaternion(input.orientation);
	geometry_msgs::PoseStamped input_s, output_s;
	input_s.header.frame_id = "map";
	input_s.pose = input;
	output_s.header.frame_id = "QR";
	
	try
	{
		input_s.header.stamp = ros::Time(0);
		input_s.header.frame_id = "map";
		listener->transformPose("QR", input_s, output_s);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
		return false;
	}

	output = output_s.pose;
  

	return true;
}


Eigen::VectorXf Visualize::FitterLeastSquareMethod(vector<float> &X, vector<float> &Y, uint8_t orders)
{
  Eigen::VectorXf result;
  // abnormal input verification
    if (X.size() < 2 || Y.size() < 2 || X.size() != Y.size() || orders < 1)
      return result;

    // map sample data from STL vector to eigen vector
    Eigen::Map<Eigen::VectorXf> sampleX(X.data(), X.size());
    Eigen::Map<Eigen::VectorXf> sampleY(Y.data(), Y.size());

    Eigen::MatrixXf mtxVandermonde(X.size(), orders + 1);  // Vandermonde matrix of X-axis coordinate vector of sample data
    Eigen::VectorXf colVandermonde = sampleX;              // Vandermonde column

    // construct Vandermonde matrix column by column
    for (size_t i = 0; i < orders + 1; ++i)
    {
        if (0 == i)
        {
            mtxVandermonde.col(0) = Eigen::VectorXf::Constant(X.size(), 1, 1);
            continue;
        }
        if (1 == i)
        {
            mtxVandermonde.col(1) = colVandermonde;
            continue;
        }
        colVandermonde = colVandermonde.array()*sampleX.array();
        mtxVandermonde.col(i) = colVandermonde;
    }

    // calculate coefficients vector of fitted polynomial
    result = (mtxVandermonde.transpose()*mtxVandermonde).inverse()*(mtxVandermonde.transpose())*sampleY;
    return result;
}

int main(int argc, char ** argv) 
{
  ROS_INFO("VISUALIZE PATH");
  ros::init(argc, argv, "visulize_node");
  Visualize MyVisualize;
  MyVisualize.Execute();

  return 0;
}
#endif
