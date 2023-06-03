#include <visible_cable_extraction.h>
#include <chrono>
#include <list>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

// debug functions

void draw_Line_Segments(cv::Mat & image, const std::list<cv::Point>& segments, cv::Scalar color){
  std::list<cv::Point>::const_iterator it_prev =  segments.begin(), it_curr;

  for(it_curr = ++it_prev; it_curr != segments.end();++it_curr){
    cv::line(image, *it_prev, *it_curr, color, 2);
    it_prev=it_curr;
  }
}

void print_Points(const std::string & name, const std::list<cv::Point>& points){
    std::cout<<name<<":";
    for(std::list<cv::Point>::const_iterator it=points.begin(); it != points.end();++it){
      std::cout<<" "<<(*it);
    }
    std::cout<<std::endl;
}


// auxiliary functions

std::vector<cv::Point2d> reOrderSampledPoints(const std::vector<cv::Point2d> sampled, const int contourPointsCloseToHoldInd, const bool reOrderDirection)
{
	std::vector<cv::Point2d> ContourReOrder;
	switch (reOrderDirection) {
		case 1:
			for (size_t i = 0; i < sampled.size() - contourPointsCloseToHoldInd; i++) {
				ContourReOrder.push_back(sampled[i + contourPointsCloseToHoldInd]);
			}
			for (size_t i = 0; i < contourPointsCloseToHoldInd; i++) {
				ContourReOrder.push_back(sampled[i]);
			}
			break;
		case 0:
			for (size_t i = 0; i <= contourPointsCloseToHoldInd; i++) {
				ContourReOrder.push_back(sampled[contourPointsCloseToHoldInd - i]);
			}
			for (size_t i = 0; i < sampled.size() - contourPointsCloseToHoldInd; i++) {
				ContourReOrder.push_back(sampled[sampled.size() - i - 1]);
			}
			break;
	}
	return ContourReOrder;
}

std::vector<cv::Point2d> select_points_by_distance(const cv::Point2d& ref_point, const cv::Point2d& ref_point_previous, const cv::Point2d& final_point,
                                                  int sampling_distance, std::list<cv::Point>& cable_border){
    std::vector<cv::Point2d> selected_points;

    while(not cable_border.empty()
          and cv::norm(ref_point - cv::Point2d(cable_border.front().x,cable_border.front().y))<= sampling_distance) {
      selected_points.push_back(cv::Point2d(cable_border.front().x, cable_border.front().y));
      cable_border.erase(cable_border.begin());
    }
    if(selected_points.empty()){//deal with the case when no point of the border is at minimal distance
      cv::Point2d next;
      if(cable_border.empty()){
        next=cv::Point2d(final_point.x,final_point.y);
      }
      else{//cable border not empty
        //remove points if distance to reference point increase
        while(not cable_border.empty()
              and cv::norm(ref_point - cv::Point2d(cable_border.front().x,cable_border.front().y)) >= cv::norm(ref_point_previous - cv::Point2d(cable_border.front().x,cable_border.front().y))){
          //avoid distance diverge
          cable_border.erase(cable_border.begin());
        }
        if(cable_border.empty()){
          next=cv::Point2d(final_point.x,final_point.y);
        }
        else{
          next= cv::Point2d(cable_border.front().x,cable_border.front().y);
        }
      }
      //now create a fake point
      selected_points.push_back(cv::Point2d(ref_point.x+(next.x - ref_point.x)/sampling_distance, ref_point.y+(next.y - ref_point.y)/sampling_distance));
    }
    return (selected_points);
}

double compute_length(const std::vector<cv::Point2d>& cable){
  cv::Point2d previous(-1.0,-1.0);
  double distance=0;
  for(std::vector<cv::Point2d>::const_iterator it=cable.begin();it != cable.end();++it){
    if(previous.x != -1){
      distance += cv::norm(*it-previous);
    }
    previous=*it;
  }
  return (distance);
}

bool circle_contains(const cv::Point& to_check, const cv::Point& center, double radius){
  return (cv::norm(to_check-center)<=radius);
}

int get_Index_Of_Common_Borders_Point(int index_begin, int index_end,
                                      const cv::Point & zone_center, double zone_radius,
                                      const std::vector<cv::Point>& final_contour){
  double min_distance=999999999.0;
  int found_index=-1;
  int points_to_check;
  if(index_begin > index_end){
    points_to_check = final_contour.size()-index_begin + index_end + 1;
  }
  else{
    points_to_check = index_end-index_begin + 1;
  }
  for(size_t i = 0; i < points_to_check; ++i){//foreach between begin and end
    int index=(i+index_begin)%final_contour.size();
    const cv::Point & c = final_contour[index];
    if(circle_contains(c, zone_center, zone_radius)){//point in the zone
        double dist = cv::norm(c-zone_center);
        if(min_distance > dist){
          min_distance =dist;
          found_index=index;
        }
    }
  }
  return (found_index);
}

std::list<cv::Point> get_All_Successive_Points_Between(const std::vector<cv::Point>& final_contour, int index_begin, int index_end, bool reverse){
    std::list<cv::Point> to_return;
    int points_to_check;
    if(index_begin > index_end){
      points_to_check = final_contour.size()-index_begin + index_end + 1;
    }
    else{
      points_to_check = index_end-index_begin + 1;
    }
    for(size_t i = 0; i < points_to_check; ++i){//foreach between begin and end
      int index=(i+index_begin)%final_contour.size();
      const cv::Point & c = final_contour[index];
      if(reverse){
        to_return.push_front(c);
      }
      else{
        to_return.push_back(c);
      }
    }
    return (to_return);
}


cv::Mat find_Largest_Connected_Area(cv::Mat binaryImg)
{
  cv::Mat out;
  cv::Mat stats;
  cv::Mat centroids;
  int label = connectedComponentsWithStats(~binaryImg, out, stats, centroids);
  // find the right connected area
  int rightLabel = 1;
  int area = stats.at<int>(1,cv::CC_STAT_AREA);
  if (label > 2)
  {
    for (size_t numOflabel = 2; numOflabel < label - 1; numOflabel++)
    {
        int areaNew = stats.at<int>(numOflabel, cv::CC_STAT_AREA);
        if (areaNew > area)
        {
          rightLabel = numOflabel;
        }
    }
  }
  cv::Mat cable;
  compare(out, rightLabel, cable, cv::CMP_EQ);  // white cable with black background
  return cable;
}


cv::Mat get_Cable_Img(const cv::Mat image, const cv::Rect2i initialROI, const int threshold, const int maskSize)
{
  cv::Mat imageGray;                              // Gray scale image
  cv::Mat cable, binarized;
  cv::cvtColor(image, imageGray, cv::COLOR_BGR2HSV);
  static cv::Mat maskCable(cv::Mat(image.rows, image.cols, CV_8UC1, 255));
  // std::cout << "Select ROI for the cable detection " << '\n';
  // Rect2i ROI = rectResize(ROIresized, resizingFactor);
  maskCable(initialROI) = 0;
  // imshow("before mask", imageGray);
  // imshow("mask", maskCable);
  bitwise_or(imageGray, maskCable, binarized); // apply mask
  // bitwise_or(imageGray, maskTips, imageTips); // apply mask
  cv::threshold(binarized, binarized, threshold, 255, cv::THRESH_BINARY_INV);
  // update the mask
  cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(maskSize, maskSize));
  cv::erode(~binarized, maskCable, structuringElement);
  return (binarized);
}

std::vector<cv::Point2d> get_Object_Contour(const cv::Mat& image, cv::Mat& blob, const int hue_value = 95, const int thres = 10, const int erosionSize = 5, const int dilationSize = 1)
{
  cv::Mat closedObjects, mask1, mask2, imageHSV;
  cv::cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);
  // int thres = 8;
  // rigid object
  // cv::inRange(imageHSV, cv::Scalar(hue_value, 100, 150), cv::Scalar(hue_value + thres, 255, 255), mask1);
  // cv::inRange(imageHSV, cv::Scalar(hue_value - thres, 100, 150), cv::Scalar(hue_value, 255, 255), mask2);
  // foam
  cv::inRange(imageHSV, cv::Scalar(hue_value, 100, 80), cv::Scalar(hue_value + thres, 255, 255), mask1);
  cv::inRange(imageHSV, cv::Scalar(hue_value - thres, 100, 80), cv::Scalar(hue_value, 255, 255), mask2);
  cv::Mat selectColor = mask1 | mask2;        // get the closed object
  cv::Mat structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(erosionSize, erosionSize));
  cv::erode(selectColor, closedObjects, structuringElement);  // remove noise
  structuringElement = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(dilationSize, dilationSize));
  cv::dilate(closedObjects, closedObjects, structuringElement);  // remove noise
  blob = find_Largest_Connected_Area(~closedObjects);
  cv::blur(blob, blob, cv::Size(3, 3));
  cv::GaussianBlur(blob, blob, cv::Size(3, 3), 0);
  // cv::Mat edges;
  // cv::Canny(blob, edges, 100, 200, 3); // find edges
  std::vector<std::vector<cv::Point> > contours;
  std::vector<cv::Vec4i> hierarchy;
  findContours(blob, contours, hierarchy,cv::RETR_TREE, cv::CHAIN_APPROX_NONE, cv::Point(0, 0));
  int indexContourMaxPoint = 0;
  for (size_t i = 0; i < contours.size(); i++) {
  if (contours[i].size() > contours[indexContourMaxPoint].size()) {
      indexContourMaxPoint = i;
    }
  }
  // convert to double
  std::vector<cv::Point2d> to_return;
  for (size_t i = 0; i < contours[indexContourMaxPoint].size(); i++) {
    to_return.push_back(static_cast<cv::Point2d>(contours[indexContourMaxPoint][i]));
  }
  to_return.push_back(to_return[0]);  // add the first point to form a closed contour for length computation(fixed bug)
  return to_return;
}


cv::Point2d create_ponderated_centroid(const std::vector<cv::Point2d>& border1, const std::vector<cv::Point2d>& border2){
  double x1 = 0, y1=0, x2 = 0, y2=0;
  for(int i=0; i < border1.size(); ++i){
    x1+=border1[i].x;
    y1+=border1[i].y;
  }
  x1/=border1.size();
  y1/=border1.size();

  for(int i=0; i < border2.size(); ++i){
    x2+=border2[i].x;
    y2+=border2[i].y;
  }
  x2/=border2.size();
  y2/=border2.size();
  return cv::Point2d((x1+x2)/2.0,(y1+y2)/2.0);
}

std::vector<cv::Point2d> generate_sampled_cable(double distance_per_sample, std::vector<cv::Point2d>& extracted_cable, std::vector<double>& segments){
  std::vector<cv::Point2d> to_return;
  double curr_distance=0.0;
  cv::Point2d previous(-1,-1);
  while(not extracted_cable.empty()){
    cv::Point2d p = extracted_cable.front();

    if(previous.x!=-1){
      //compute the distance
      double dist = cv::norm(p-previous);
      if(curr_distance+dist <= distance_per_sample){
        extracted_cable.erase(extracted_cable.begin());
        curr_distance+=dist;
        previous=cv::Point2d(p.x,p.y);
      }
      else{//curr_distance+dist > distance_per_sample
        double needed_distance_ratio = (distance_per_sample - curr_distance)/dist;
        cv::Point2d created = cv::Point2d(previous.x + ((double)p.x-previous.x)*needed_distance_ratio, previous.y + (p.y-previous.y)*needed_distance_ratio);
        to_return.push_back(created);
        segments.push_back(curr_distance+cv::norm(created-previous));
        previous=created;
        curr_distance=0.0;
      }
    }
    //adding the first point
    else{
      extracted_cable.erase(extracted_cable.begin());
      previous=cv::Point2d(p.x,p.y);
      to_return.push_back(previous);
    }

  }
  if(curr_distance+0.001 > distance_per_sample){
    to_return.push_back(previous);//adding the last point
    segments.push_back(curr_distance);
    curr_distance=0.0;
  }
  return (to_return);
}

// function defined by the API
std::vector<cv::Point2d> extract_Sampled_Cable_From_Image(cv::Mat& image,
    int sampling_distance, int cable_points, bool debug_log,
    const cv::Point& holding_point, int radius_hold,
    const cv::Point& ending_point, int radius_end, const cv::Rect2i initialROI){

  std::vector<cv::Point2d> sampled_cable;
  //binarize, remove noise and get only the cable in image
  auto start_cable_img = std::chrono::high_resolution_clock::now();
  cv::Mat binarized = get_Cable_Img(image, initialROI, 70, 10);
  auto end_cable_img = std::chrono::high_resolution_clock::now();

  cv::Mat bin_to_print;
  if(debug_log){
    cv::cvtColor(binarized, bin_to_print, cv::COLOR_GRAY2BGR);
    cv::circle(bin_to_print, holding_point, radius_hold, cv::Scalar(0,255,255),1);
    cv::circle(bin_to_print, ending_point, radius_end, cv::Scalar(255,0,255),1);
  }
  // cv::erode(image, image, structuringElement);  // update the mask
  //step 4 : finding contour
  //get the contour as a sequence of points
  std::vector<std::vector<cv::Point>> contours;
  std::vector<cv::Point> final_contour;
  std::vector<cv::Vec4i> hierarchy;
  cv::findContours(binarized, contours, hierarchy, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE );

  int start_point=-1;
  //select the contour containing hold point
  for(size_t idx = 0; idx < contours.size(); idx++)//foreach contour
  {
    for(size_t idx_p = 0; idx_p < contours[idx].size(); ++idx_p){//foreach point of this contour
      const cv::Point & c = contours[idx][idx_p];
      if(circle_contains(c, holding_point, radius_hold)){//if on point of this contour belongs to the start zone
        for(size_t idx_p2 = 0; idx_p2 < contours[idx].size(); ++idx_p2){//foreach point of the same contour
          const cv::Point & c2 = contours[idx][idx_p2];
          if((c2.x != c.x or c.y != c2.y) //not the same point
            and circle_contains(c2, ending_point, radius_end)){//if on point of this contour belongs to the start zone
            final_contour = contours[idx];
            start_point=idx_p;//memorize the first point to start with
            break;
          }
        }
        //OK this contour is selected because hold point found or not so immediately check another contour
        break;
      }
    }
    if(not final_contour.empty()){
      break;
    }
  }
  if(final_contour.empty()){
    if(debug_log){
      std::cout<<" NO CABLE FOUND -> printing detected contours"<<std::endl;
      drawContours(bin_to_print, contours, -1, cv::Scalar(255,0,255));
      imshow("binarized", bin_to_print);
    }
    return (sampled_cable);//return empty vector to say "nothing found"
  }

  // step 5 getting the two cable borders vectors from the cable contour, starting from hold_point_area and ending in ending_point_area
  // step 5.1 getting index of start and end points for each border.
  enum{
    IN_HOLDING_ZONE,
    IN_ENDING_ZONE,
    FILLING_BORDER_1,
    FILLING_BORDER_2,
    FINISHED
  };
  int state=IN_HOLDING_ZONE;
  int start_1=-1;
  int start_2=-1;
  int end_1=-1;
  int end_2=-1;
  int prev_p=-1;
  for(size_t i = 0; i < final_contour.size(); ++i){//foreach point of the final contour
    int index=(i+start_point)%final_contour.size();
    const cv::Point & c = final_contour[index];
    switch(state){
    case IN_HOLDING_ZONE:{
      if(not circle_contains(c, holding_point, radius_hold)){//if point of the contour DOES NO MORE belongs to the start zone
        //this means we start filling the FIRST border
        state=FILLING_BORDER_1;
        start_1 = prev_p;//first point at the limits of the circle
      }
      else{
        prev_p=index;
      }
    }
    break;
    case IN_ENDING_ZONE:{
      if(not circle_contains(c, ending_point, radius_end)){//if point of the contour DOES NO MORE belongs to the end zone
        //this means we start filling the SECOND border
        state=FILLING_BORDER_2;
        end_2=prev_p;
      }
      else{
        prev_p=index;
      }
    }
    break;
    case FILLING_BORDER_1:{
      if(circle_contains(c, ending_point, radius_end)){//if point of the contour DOES belongs to the end zone
        //this means we finished filling the first border
        state=IN_ENDING_ZONE;
        end_1=index;
      }
    }
    break;
    case FILLING_BORDER_2:{
      if(circle_contains(c, holding_point, radius_hold)){//if point of the contour DOES belongs to the start zone
        state=FINISHED;
        start_2=index;
      }
    }
    break;
    }
    if(state==FINISHED){
      break;
    }
  }
  if(state!=FINISHED){//we made a complete loop without finding a point of border 2 inside the holding area
    start_2 = prev_p;
  }

  //step 5.2 : we have extremities for the two borders but now we want the common starting and ending points.
  int start_index = get_Index_Of_Common_Borders_Point(end_2, start_1,holding_point, radius_hold, final_contour);
  int end_index = get_Index_Of_Common_Borders_Point(end_1, start_2,ending_point, radius_end, final_contour);
  // if(debug_log){
  //   std::cout<<"start zone index = "<<start_index<<" point="<<final_contour[start_index]<<std::endl;
  //   std::cout<<"end zone index = "<<end_index<<" point="<<final_contour[end_index]<<std::endl;
  // }
  //step 5.3 : finally extract the two borders
  std::list<cv::Point> cable_border_1 =  get_All_Successive_Points_Between(final_contour, start_index, end_index, false);
  std::list<cv::Point> cable_border_2 =  get_All_Successive_Points_Between(final_contour, end_index, start_index, true);//build in reverse order

  if(debug_log){
    draw_Line_Segments(bin_to_print, cable_border_1, cv::Scalar(0,120,120));
    draw_Line_Segments(bin_to_print, cable_border_2, cv::Scalar(120,0,120));
    // print_Points("border1", cable_border_1);
    // print_Points("border2", cable_border_2);
  }

  //step 6 compute the one dimensional space of the cable
  std::vector<cv::Point2d> extracted_cable;
  //use the first point in each border
  cv::Point2d extracted_begin_point=cv::Point2d(((double)cable_border_1.front().x+(double)cable_border_2.front().x)/2.0, ((double)cable_border_1.front().y+(double)cable_border_2.front().y)/2.0);
  extracted_cable.push_back(extracted_begin_point);
  cable_border_1.pop_front();
  cable_border_2.pop_front();
  cv::Point2d extracted_end_point=cv::Point2d(((double)cable_border_1.back().x+(double)cable_border_2.back().x)/2.0, ((double)cable_border_1.back().y+(double)cable_border_2.back().y)/2.0);
  cable_border_1.pop_back();
  cable_border_2.pop_back();

  std::vector<cv::Point2d> border1_selected;
  std::vector<cv::Point2d> border2_selected;
  cv::Point2d previous_ref_point = extracted_cable.back();//initially previous ref point == ref point
  while(not cable_border_2.empty() or not cable_border_1.empty()){
    const cv::Point2d& ref_point = extracted_cable.back();
    border1_selected = select_points_by_distance(ref_point, previous_ref_point, extracted_end_point, sampling_distance, cable_border_1);
    border2_selected = select_points_by_distance(ref_point, previous_ref_point, extracted_end_point, sampling_distance, cable_border_2);
    extracted_cable.push_back(create_ponderated_centroid(border1_selected,border2_selected));
    previous_ref_point = ref_point;
  }
  // if(debug_log){
  //   std::cout<<"extracted_cable points extracted= "<<extracted_cable.size()<<std::endl;
  // }
  // //step 7 sampleing the cable
  //
  std::vector<double> distances;
  double cable_length= compute_length(extracted_cable);
  double distance_between_points = cable_length/(double)(cable_points-1);//distance * number of segments = total cable length (with number of segments == nb points -1)
  sampled_cable = generate_sampled_cable(distance_between_points, extracted_cable, distances);

  if(debug_log){
    // std::cout<<"nb points extracted= "<<sampled_cable.size()<<" desired number of points="<<cable_points<<" cable length="<<cable_length<<" desired distance between points="<<distance_between_points<<"):";
    // for(int i =0; i< sampled_cable.size();++i){
    //   if(i>0){
    //     std::cout<<" -> ("<<cv::norm(sampled_cable[i]-sampled_cable[i-1])<<")";//for indication since this distance is measured in 2d space
    //   }
    //   std::cout<<" "<<sampled_cable[i];
    // }
    // std::cout<<std::endl;
    // std::cout<<distances.size()<<" segments with distances in cable space:";
    // for(int i =0; i< distances.size();++i){
    //   std::cout<<" "<<distances[i];
    // }
    // std::cout<<std::endl;

    imshow("binarized", bin_to_print);
  }

  return(sampled_cable);
}

std::vector<cv::Point2d> extract_Closed_Contour_From_Image(cv::Mat& image,
    int sampling_distance, int cable_points, bool debug_log,
    const cv::Point& holding_point, int radius_hold)
{
  cv::Point2d holding_point2d(double(holding_point.x), double(holding_point.y));
  cv::Mat imageHSV, blob;
  std::vector<double> distances;
  cvtColor(image, imageHSV, cv::COLOR_BGR2HSV);
  std::vector<cv::Point2d> coutour_points = get_Object_Contour(image, blob);
  cv::Moments mObj = cv::moments(~blob);
  cv::Point2d centroid(mObj.m10/mObj.m00, mObj.m01/mObj.m00);
  double contour_length= compute_length(coutour_points);
  double distance_between_points = contour_length/(double)(cable_points-1);
  std::vector<cv::Point2d> sampled = generate_sampled_cable(distance_between_points, coutour_points, distances);
  sampled.pop_back(); // delete the duplicate point
  cv::Point2d contourPointsCloseToHold;
    int contourPointsCloseToHoldInd;
    double distance = 9999999;
    for (size_t i = 0; i < sampled.size(); i++) {
      cv::Point2d pointOnContour(sampled[i].x, sampled[i].y);
      if (cv::norm(pointOnContour - holding_point2d) < distance) {
        contourPointsCloseToHold = pointOnContour;
        contourPointsCloseToHoldInd = i;
        distance = cv::norm(pointOnContour - holding_point2d);
      }
    }
    int numOfPointSkipped = 2;
    cv::Point2d baseVector = holding_point2d - centroid;
    int previousPointInd = contourPointsCloseToHoldInd - numOfPointSkipped;
    if (previousPointInd < 0) {
      previousPointInd = sampled.size() + contourPointsCloseToHoldInd - numOfPointSkipped - 1;
    }
    int nextPointInd = contourPointsCloseToHoldInd + numOfPointSkipped;
    if (nextPointInd > sampled.size() - 1) {
      nextPointInd = contourPointsCloseToHoldInd - sampled.size() + numOfPointSkipped + 1;
    }
    cv::Point2d previousVector = sampled[previousPointInd] - centroid;
    cv::Point2d nextVector = sampled[nextPointInd] - centroid;
    bool reOrderDirection;
    switch (baseVector.cross(previousVector) > 0) {
      case 1:
        reOrderDirection = 0;
      break;
      case 0:
        reOrderDirection = 1;
      break;
    }
    // re-arrange the points
    cv::Mat bin_to_print;
    if (debug_log == true) {

    }
    std::vector<cv::Point2d> ContourReOrder = reOrderSampledPoints(sampled, contourPointsCloseToHoldInd, reOrderDirection);
    return ContourReOrder;
}


//utility function provided to API user to extract a speicifc point with a specific color
bool get_colored_zone(const cv::Mat& rgb_image, int reference_hsv_color_value, int threshold, int radius, cv::Point& zone_center, cv::Rect & result_zone){
  cv::Mat hsv;
  cvtColor(rgb_image, hsv, cv::COLOR_BGR2HSV);
  cv::Mat mask_upper, mask_lower;
  bool upper_ok=false, lower_ok=false;
  // For green color detection
  if(reference_hsv_color_value+threshold <= 180){
    // cv::inRange(hsv, cv::Scalar(reference_hsv_color_value, 50, 50), cv::Scalar(reference_hsv_color_value+threshold, 255, 255), mask_lower);
    cv::inRange(hsv, cv::Scalar(reference_hsv_color_value, 50, 200), cv::Scalar(reference_hsv_color_value+threshold, 255, 255), mask_lower);
    lower_ok=true;
  }
  if(reference_hsv_color_value-threshold >= 0){
    // cv::inRange(hsv, cv::Scalar(reference_hsv_color_value-threshold, 100, 100), cv::Scalar(reference_hsv_color_value, 255, 255), mask_upper);
    cv::inRange(hsv, cv::Scalar(reference_hsv_color_value-threshold, 100, 200), cv::Scalar(reference_hsv_color_value, 255, 255), mask_upper);
    upper_ok=true;
  }
  if(upper_ok and lower_ok){
    hsv = mask_upper | mask_lower;        // With this red will be white and rest will be black
  }
  else if(upper_ok){
    hsv = mask_upper;
  }
  else if(lower_ok){
    hsv = mask_lower;
  }
  else{
    return false;
  }
  cv::Moments m = moments(hsv,true);
  zone_center = cv::Point(m.m10/m.m00, m.m01/m.m00);
  int val_x = (zone_center.x-radius)<0?0:zone_center.x-radius;
  int val_y = (zone_center.y-radius)<0?0:zone_center.y-radius;
  int val_width = ((val_x + 2*radius) > rgb_image.cols)?(rgb_image.cols-val_x):(2*radius);
  int val_height = ((val_y + 2*radius) > rgb_image.rows)?(rgb_image.rows-val_y):(2*radius);
  result_zone = cv::Rect(val_x, val_y,val_width, val_height);
  // std::cout<<"x="<<zone_center.x<<" y="<<zone_center.y<<std::endl;
  // std::cout<<"top left.x="<<val_x<<" top left.y="<<val_y<<" top left.width="<<val_width<<" top left.height="<<val_height<<std::endl;
  return true;
}

//utility function provided to API user to get the area for a fixed point and radius
cv::Rect zone_From_Center(const cv::Mat& rgb_image, const cv::Point zone_center, int radius){
  int val_x = (zone_center.x-radius)<0?0:zone_center.x-radius;
  int val_y = (zone_center.y-radius)<0?0:zone_center.y-radius;
  int val_width = ((val_x + 2*radius) > rgb_image.cols)?(rgb_image.cols-val_x):(2*radius);
  int val_height = ((val_y + 2*radius) > rgb_image.rows)?(rgb_image.rows-val_y):(2*radius);
  cv::Rect result_zone = cv::Rect(val_x, val_y,val_width, val_height);
  return result_zone;
}
