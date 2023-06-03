# -*- coding: utf-8 -*-
"""
Created on Mon Mar 15 12:28:12 2021

"""

import cv2 as cv                                
import numpy as np                        
import matplotlib.pyplot as plt       
from matplotlib.colors import hsv_to_rgb    
import pyrealsense2 as rs
from time import process_time  
#import open3d as o3d

from functions_display import *


def img_segmentation(hsv_pic, lower, upper, nb_op, nb_close):
    
    mask = cv.inRange(hsv_pic, lower, upper)
    # plt.imshow(mask)
    # plt.show()
    
    ### MORPHOLOGICAL TRANSFO
    kernel = np.ones((5,5),np.uint8)
    opening = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=nb_op)
    closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=nb_close)
    
    # plt.imshow(closing)
    # plt.show()
    
    return closing



def get_Holding_data(mask, radius, color_shape):
    ''' returns holding point and ROI around it'''
    # calculate moments of binary image
    M = cv.moments(mask)
    holding_point = np.array([[int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])]])
    
    val_x = 0 if (holding_point[0][0] - radius)<0 else holding_point[0][0] - radius
    val_y = 0 if (holding_point[0][1] - radius)<0 else holding_point[0][1]- radius
    val_width = (color_shape[1]-val_x) if ((val_x + 2*radius) > color_shape[1]) else (2*radius)
    val_height = (color_shape[0]-val_y) if ((val_y + 2*radius) > color_shape[0]) else (2*radius)
    # result_zone = np.array([[val_x, val_x+val_height, val_x+val_height, val_x],
    #                         [val_y, val_y, val_y+val_width, val_y+val_width]]) ## 4points of the rectangle
    result_zone = np.array([[val_x, val_x+val_height, val_x],
                            [val_y, val_y, val_y+val_width]]) ## 3points of the rectangle
    
    ##Displaying the image 
    # plt.imshow(mask)
    # plt.plot(result_zone[0],result_zone[1])
    # plt.scatter(holding_point[0], holding_point[1], color='red')
    
    return holding_point, result_zone
    



def point_cloud(depth, intrinsics):
    """Transform a depth image into a point cloud with one point for each
    pixel in the image, using the camera transform for a camera
    centered at cx, cy with field of view fx, fy.
    """
    fx, fy = intrinsics.fx, intrinsics.fy
    cx, cy = intrinsics.ppx, intrinsics.ppy
    
    mask = np.where(depth > 0)
    
    x = mask[1]
    y = mask[0]
    
    world_x = (x -cx) * depth[y, x] / fx
    world_y = -(y - cy) * depth[y, x] / fy
    world_z = -depth[y, x]

    return np.vstack((world_x, world_y, world_z)).T 

def outlier_removal(pc, coef_limit, nb_it):
    
    # cv.imshow("outlier removal",pc2img(pc))
    # cv.waitKey(0)

    for i in range(nb_it):
        mean = np.mean(pc, axis = 0)
        distances = np.linalg.norm(pc - mean, axis = 1)
        limit_distance = np.std(distances)
        pc = pc[distances < (coef_limit)*limit_distance]
        # cv.imshow("outlier removal",pc2img(pc))
        # cv.waitKey(0)
        
    return pc

def sub_sampling(pc, nb_points):
    step = round(pc.shape[0]/nb_points) -1
    pc_reduced = np.zeros((nb_points,pc.shape[1]))
    pc_reduced[0] = pc[0]
    for i in range(1, nb_points):
        pc_reduced[i] = pc[i * step]
    
    return pc_reduced



def find_holding_point(pc, holding_point2d):
    idx = np.linalg.norm(pc - holding_point2d, axis = 1).argmin()    
    return pc[idx], idx


        
def reOrderSampledPoints(pc, PointCloseToHoldInd, reOrderDirection):
    if reOrderDirection:
        re_order = np.vstack((pc[PointCloseToHoldInd:],pc[:PointCloseToHoldInd]))
    else:
        re_order = np.vstack((pc[PointCloseToHoldInd:0:-1], pc[0], pc[-1:PointCloseToHoldInd:-1]))
    
    return re_order

# ----------------------------------------------------------------------------
## ENTIRE PROCESS TO GET POINT CLOUD

def image_processing(color_frame, aligned_depth_frame, depth_scale, nb_points, radius_holding =10):
    ''' returns point cloud (only 2D) from frames'''
    # color_frame = frames.get_color_frame()
    # aligned_depth_frame = frames.get_depth_frame()

    # start = process_time()  
    
    ## FRAMES DATAS --------------------------
    color = np.asanyarray(color_frame.get_data())
    color_shape = color.shape
    
    aligned_depth = np.asanyarray(aligned_depth_frame.get_data())
    
    ##Show the two frames together:
    #colorizer = rs.colorizer()
    #colorized_depth = np.asanyarray(colorizer.colorize(aligned_depth_frame).get_data())
    #images = np.hstack((color, colorized_depth))
    # plt.figure()
    # plt.imshow(color)
    # plt.show()
    
    # stop1 = process_time()  
    # print('aligned rgb/depth capture:',stop1-start)
    
    ## IMAGE SEGMENTATION --------------------
    hsv_pic = cv.cvtColor(color, cv.COLOR_RGB2HSV)
    
    ## threshold
    light_blue = (205/2, 100*255/100, 100*255/100) 
    dark_blue = (165/2, 50*255/100, 10*255/100)
    
    # rgb_light = np.uint8([[[21, 145, 178]]])
    # rgb_dark = np.uint8([[[0, 33, 46]]])
    
    # light_blue = cv.cvtColor(rgb_light, cv.COLOR_RGB2HSV)
    # dark_blue = cv.cvtColor(rgb_dark, cv.COLOR_RGB2HSV)
    
    # lo_square = np.full((10, 10, 3), cv.cvtColor(light_blue, cv.COLOR_HSV2RGB), dtype=np.uint8) / 255.0
    # do_square = np.full((10, 10, 3), cv.cvtColor(dark_blue, cv.COLOR_HSV2RGB), dtype=np.uint8) / 255.0
    
    mask = img_segmentation(hsv_pic, dark_blue, light_blue, 7, 7)
    
    if not mask.any():
        print("image segmentation failed")
        return np.zeros((nb_points, 2))

    ### PARAMETERS ----------------------------------
    holding_point, resulting_zone = get_Holding_data(mask, radius_holding, color_shape)

    # stop2 = process_time()  
    # print('image segmentation:',stop2-stop1)
    
    ## POINT CLOUD ------------------------------
    intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
    # print(intrinsics)
    
    depth = cv.bitwise_and(aligned_depth, aligned_depth, mask=mask)*depth_scale

    pc = point_cloud(depth, intrinsics)
    
    #projects holding point
    x = holding_point[0][0]
    y = holding_point[0][1]
    holding_point_world = np.array([[(x -intrinsics.ppx) * depth[y, x] / intrinsics.fx,
                                    -(y - intrinsics.ppy) * depth[y, x] / intrinsics.fy]])
    #projects holding zone
    # x = resulting_zone[0]
    # y = resulting_zone[1]
    # resulting_zone_world = np.vstack(((x -intrinsics.ppx) * depth[y, x] / intrinsics.fx,
    #                                 -(y - intrinsics.ppy) * depth[y, x] / intrinsics.fy))
    
    # stop3 = process_time()  
    pc = pc[:,:2] #only x and y
    # print('point cloud:', stop3-stop2)
    
    # OUTLIER REMOVAL-------------------------------
    pc = outlier_removal(pc, 2.4, 1)

    # stop4 = process_time()  
    # print('outlier removal:', stop4-stop3)

    # pc = pc[np.random.choice(len(pc),3*nb_points,replace=False),:]
    # pc = sub_sampling(pc, 3*nb_points)

    # pc = outlier_removal(pc, 2.2, 1)
    
    # DOWN SAMPLING -------------------------------
    # pc_reduced = sub_sampling(pc, nb_points)
    pc_reduced = pc[np.random.choice(len(pc),nb_points,replace=False),:]
    
    # stop5 = process_time()  
    # print('down sampling:', stop5-stop4)
    
    ## FIGURE ----------------------------------------
    # pcd = o3d.geometry.PointCloud()
    # pcd.points = o3d.utility.Vector3dVector(pc_reduced)
    # o3d.visualization.draw_geometries([pcd])

    ## ------------------------------------------##
    #             2D POINT CLOUD                 ##
    ## ------------------------------------------##
    
    # HOLDING POINT  -------------------------------  
    PointCloseToHold, PointCloseToHoldInd = find_holding_point(pc_reduced, holding_point_world)
    
    # REORDER --------------------------------------
    reOrderDirection= True
    pc_reordered = reOrderSampledPoints(pc_reduced, PointCloseToHoldInd, reOrderDirection)
    
    
    # stop_end = process_time()  
    # print('total process time:', stop_end - start)
    
    return pc_reordered #,resulting_zone_world

def contour_length(cnt):
    length = 0
    prev = cnt[0]
    cnt = cnt[1:,:] #delete first element
    
    for i in cnt:
        length += np.linalg.norm(i - prev)
        prev = i
        
    return length
        

def sample_contours(cnt, delta_dist):
    curr_dist = 0 
    prev = cnt[0]
    sampled = prev
    cnt = cnt[1:,:] #delete first element
    
    while cnt.any():
        p = cnt[0]
        dist = np.linalg.norm(p -prev)
        if (curr_dist + dist) <= delta_dist:
            cnt = cnt[1:,:]
            curr_dist += dist
            prev = p
        else:
            ratio = (delta_dist-curr_dist)/dist
            new_point = np.array([[prev[0]+ (p[0] - prev[0])*ratio, prev[1]+ (p[1] - prev[1])*ratio]]).reshape((2,))
            sampled = np.vstack((sampled, new_point))
            curr_dist = 0
            prev = new_point
            
    #last point
    if curr_dist+0.001> delta_dist:
         sampled = np.vstack((sampled, prev))
         
    return sampled
        

def reOrder_center(pc, center):
    dist = np.arctan2(pc[:,1]-center[0,1], pc[:,0] - center[0,0])
    idx = np.argsort(dist) #get indexes of closest points to center
    idxs = np.vstack((idx,idx)).transpose()
    pc = np.take_along_axis(pc, idxs, axis = 0) #reorder pc from closest to farest
    
    return pc    

def get_center(mask):
    ''' returns the center of the object '''
    # calculate moments of binary image
    M = cv.moments(mask)
    center = np.array([[int(M["m10"] / M["m00"]),int(M["m01"] / M["m00"])]])
    
    return center
    

def image_processing_contour(color_frame, nb_points):
    ''' returns sampled contour from frames'''
    radius = 10

    ## FRAMES DATAS --------------------------
    color = np.asanyarray(color_frame.get_data())
    color_shape = color.shape

    
    ## IMAGE SEGMENTATION --------------------
    hsv_pic = cv.cvtColor(color, cv.COLOR_RGB2HSV)
    
    ## threshold
    light_blue = (205/2, 100*255/100, 100*255/100) 
    dark_blue = (165/2, 50*255/100, 10*255/100)
    
    mask = img_segmentation(hsv_pic, dark_blue, light_blue, 4, 4)
    
    # clr_bgr = cv.cvtColor(color, cv.COLOR_RGB2BGR)
    # mask_clr = cv.bitwise_and(clr_bgr, clr_bgr, mask=mask)
    # cv.imshow("mask", mask_clr)
    # cv.waitKey(1)
    # cv.imshow("color", clr_bgr)
    # cv.waitKey(1)

    if not mask.any():
        print("image segmentation failed")
        return np.zeros((nb_points, 2))

    ### PARAMETERS ----------------------------------
    contours,_ = cv.findContours(mask, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
    
    cnt = np.vstack(contours).squeeze() #convert to coordinates array
    cnt = np.vstack((cnt, cnt[0])) #add first point for closed contours
    
    #FIGURE
    # img = cv.cvtColor(color, cv.COLOR_RGB2BGR)
    # cv.drawContours(img, [cnt], 0, (0,255,0), 3)
    # cv.imshow('Contours', img)
    # cv.waitKey(0)
    # cv.destroyAllWindows()
    
    cable_length = contour_length(cnt)
    delta_dist = cable_length/(nb_points-1)
    
    sampled = sample_contours(cnt, delta_dist)

    holding_point,_ = get_Holding_data(mask, radius, color_shape)
    # HOLDING POINT  -------------------------------  
    PointCloseToHold, PointCloseToHoldInd = find_holding_point(sampled, holding_point)
    
    # REORDER --------------------------------------
    reOrderDirection= True
    pc_reordered = reOrderSampledPoints(sampled, PointCloseToHoldInd, reOrderDirection)
    

    
    return pc_reordered
