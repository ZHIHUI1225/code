import math
import numpy as np

def cos_multiple_angle(x, n):
    result = 0
    sign = 1  # Sign for alternating terms

    for k in range(0, n//2 + 1):
        term = sign * math.comb(n, 2*k) * math.cos(x)**(n - 2*k)
        result += term
        sign *= -1

    return result

def sin_multiple_angle(x, n):
    result = 0
    sign = 1  # Sign for alternating terms

    for k in range(0, n//2 + 1):
        term = sign * math.comb(n, 2*k+1) * math.sin(x) * math.cos((n - (2*k+1)) * x)
        result += term
        sign *= -1
    return result

def distance_point_to_segment(point, segment_start, segment_end):
    # 计算线段的向量
    segment_vector = segment_end - segment_start

    # 计算从起点到点的向量
    point_vector = point - segment_start

    # 计算线段长度的平方
    segment_length_sq = np.dot(segment_vector, segment_vector)

    # 如果线段长度的平方接近于0，则点和线段重合
    if segment_length_sq < 1e-6:
        return np.linalg.norm(point_vector)

    # 计算点到线段的投影比例
    projection_ratio = np.dot(point_vector, segment_vector) / segment_length_sq

    # 如果投影比例小于0，则点在线段起点之前
    if projection_ratio < 0:
        return np.linalg.norm(point_vector), segment_start

    # 如果投影比例大于1，则点在线段终点之后
    if projection_ratio > 1:
        return np.linalg.norm(point - segment_end), segment_end

    # 计算点到线段的垂直距离
    perpendicular_distance = np.linalg.norm(point_vector - projection_ratio * segment_vector)
    closest=  segment_start+ segment_vector*projection_ratio
    return perpendicular_distance, closest

def distance_point_to_polygon(point, polygon):
    min_distance = math.inf
    # 遍历多边形的边
    for i in range(len(polygon)):
        segment_start = polygon[i]
        segment_end = polygon[(i + 1) % len(polygon)]
        [distance, closest] = distance_point_to_segment(point, segment_start, segment_end)
        if distance < min_distance:
            min_distance = distance
            min_point=closest

    return min_distance,min_point


def get_sf_max_of_polygon(point, polygon):
    # point polygon array
    [d,p]=distance_point_to_polygon(point,polygon)
    [angle,idx_v1,idx_v2]=max_angle(point,polygon)
    #get Sf_max
    Sf_max=calculate_angle(p, polygon[idx_v1], polygon[idx_v2])
    # calulate K and b for cos(k\theta+b)
    K=int(3*math.pi/Sf_max)
    b=math.pi-K*Sf_max
    return Sf_max,K,b
    
def get_angle_infromation(theta,K,b,Sf):
    # Cos=cos_multiple_angle(theta, K)*math.cos(b)-sin_multiple_angle(theta,K)*math.sin(b)
    
    if K*theta+b>=math.pi:
        Cos=1
    elif K*theta+b<0:
        Cos=-1
    else:
        Cos=cos_multiple_angle(theta, K)*math.cos(b)-sin_multiple_angle(theta,K)*math.sin(b)

    return Cos



def calculate_angle(point, vertex1, vertex2):
    # 计算给定点与两个顶点形成的向量之间的夹角
    x1 = vertex1[0] - point[0]
    y1 = vertex1[1] - point[1]
    x2 = vertex2[0] - point[0]
    y2 = vertex2[1] - point[1]
    dot_product = x1*x2 + y1*y2
    norm_a = math.sqrt(x1**2 + y1**2)
    norm_b = math.sqrt(x2**2 + y2**2)
    if dot_product / (norm_a * norm_b)>1:
        theta=0
    elif dot_product / (norm_a * norm_b)<-1:
        theta=math.pi
    else:
        theta= math.acos(dot_product / (norm_a * norm_b))

    return theta

def max_angle(point, polygon):
    # 计算一个点与多边形的顶点的最大夹角
    max_angle = -math.inf
    idx_v1=0
    idx_v2=1
    for i in range(len(polygon)-1):
        for j in range(i+1,len(polygon)):
            angle = calculate_angle(point, polygon[i], polygon[(j)%len(polygon)])
            if angle > max_angle:
                max_angle = angle
                idx_v1=i
                idx_v2=(j)%len(polygon)
    return max_angle,idx_v1,idx_v2

    
# 测试
point = (3,3)
polygon = [(0, 0), (2, 0), (2, 2), (0, 2)]
[theta,p]=distance_point_to_polygon(np.array(point), np.array(polygon))
[angle,idx_v1,idx_v2]=max_angle(np.array(point), np.array(polygon))
[Sf,K,b]=get_sf_max_of_polygon(np.array(point), np.array(polygon))
cos=get_angle_infromation(angle,K,b,Sf)
print(Sf)
print(math.degrees(angle))
print(cos)
print(p)