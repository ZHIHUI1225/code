#pragma once

#include <vector>
#include <opencv2/core.hpp>

std::vector<cv::Point2d> extract_Sampled_Cable_From_Image(cv::Mat& image,
    int sampling_distance, int cable_points, bool debug_log,
    const cv::Point& holding_point, int radius_hold,
    const cv::Point& ending_point, int radius_end, const cv::Rect2i initialROI);

std::vector<cv::Point2d> extract_Closed_Contour_From_Image(cv::Mat& image,
    int sampling_distance, int cable_points, bool debug_log,
    const cv::Point& holding_point, int radius_hold);

bool get_colored_zone(const cv::Mat& rgb_image,
          int reference_hsv_color_value, int threshold,
          int radius, cv::Point& zone_center, cv::Rect & result_zone);

cv::Rect zone_From_Center(const cv::Mat& rgb_image, const cv::Point zone_center, int radius);
