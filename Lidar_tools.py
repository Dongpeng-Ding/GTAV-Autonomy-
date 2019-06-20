import numpy as np
import cv2
import math

def edge_detect(lidar_img):
    #lidar_img = lidar_img / np.max(lidar_img) * 255
    edge_y = cv2.Sobel(lidar_img, cv2.CV_64F, 0, 2, ksize=1)
    edge_x = cv2.Sobel(lidar_img, cv2.CV_64F, 2, 0, ksize=1)
    #edge = np.append(np.expand_dims(edge_y, axis=2), np.expand_dims(edge_x, axis=2), axis=2)
    #edge = np.max(edge, axis=2)
    return edge_y, edge_x

def polar_to_cartesian(rho, theta, phi):
    x = rho * np.sin(theta) * np.sin(phi)
    y = rho * np.cos(theta)
    z = rho * np.sin(theta) * np.cos(phi)
    return x, y, z


def find_lidar_theta_phi_from_image(pixel_h, pixel_v, focus, half_res_h, half_res_v): # !!! half_res is actually the pixel coord in the camera center
    # (lidar and camera are aligned in same coordinate)

    # horizontal angle to center at horizon
    pixel_x_refer2center = half_res_h - pixel_h - 0.5  # +: left, -: right

    # vertical angle to center at center vertical line
    pixel_y_refer2center = pixel_v - half_res_v + 0.5  # -: above, +: under

    theta = math.atan(pixel_y_refer2center / math.sqrt(focus ** 2 + pixel_x_refer2center ** 2)) + np.pi / 2
    phi = math.atan(pixel_x_refer2center / focus)

    return theta, phi


def find_lidar_phi_from_coord(x, x_sample, x_sample_deg):
    if x + 1 <= x_sample / 2:  # left side
        phi = (x_sample_deg * (x_sample / 2 - x) - x_sample_deg / 2) / 180 * np.pi
    else:  # right side
        phi = - (x_sample_deg * (x + 1 - x_sample / 2) - x_sample_deg / 2) / 180 * np.pi
    return phi


def find_lidar_theta_from_coord(y, y_sample_deg, upper_lim):
    theta = (upper_lim + y * y_sample_deg) / 180 * np.pi
    return theta


def find_lidar_theta_phi_from_coord_Ma(row_lidar, col_lidar, x_sample, x_sample_deg, y_sample_deg, upper_lim):
    # same as the function above, but use matrix
    theta = (upper_lim + row_lidar * y_sample_deg) / 180 * np.pi

    phi = np.where(col_lidar <= (x_sample // 2 - 1),  # left side, else right side
                   (x_sample_deg * (x_sample / 2 - col_lidar) - x_sample_deg / 2) / 180 * np.pi,
                   - (x_sample_deg * (col_lidar + 1 - x_sample / 2) - x_sample_deg / 2) / 180 * np.pi)

    return theta, phi


#def single_coord_roll_move(x, y, z, trans_x, trans_y, trans_z, rotation_matrix_pitch_roll, rotation_matrix_yaw, mode):
#    # roll and move the axis not the points ### moved to tools
#    if mode == 'move_first':
#        coord_centered = np.array([[x - trans_x], [y - trans_y], [z - trans_z]])
#        new_coord = np.dot(np.linalg.inv(rotation_matrix_yaw.T), coord_centered)
#        new_coord = np.dot(rotation_matrix_pitch_roll.T, new_coord)
#    elif mode == 'turn_first':
#        coord = np.array([[x], [y], [z]])
#        new_coord = np.dot(rotation_matrix_pitch_roll.T, coord)
#        new_coord = np.dot(np.linalg.inv(rotation_matrix_yaw.T), new_coord)
#        new_coord[0] -= trans_x
#        new_coord[1] -= trans_y
#        new_coord[2] -= trans_z
#    else:
#        print('wrong mode')
#        assert(0)
#    return new_coord[0], new_coord[1], new_coord[2]


# interpolation the R_1 at angle_1, witch between angle_0 and angle_2
def interpolation(angle_0, angle_1, angle_2, R_0, R_2):
    point1_x = R_0 * np.cos(angle_0)
    point1_y = R_0 * np.sin(angle_0)

    point2_x = R_2 * np.cos(angle_2)
    point2_y = R_2 * np.sin(angle_2)

    k = (point2_y - point1_y) / (point2_x - point1_x)

    # angle between the object surface and lidar light:
    #angle_check_0 = np.arctan(abs((k - np.tan(angle_0)) / (1 + k * np.tan(angle_0))))
    angle_check_1 = np.arctan(abs((k - np.tan(angle_1)) / (1 + k * np.tan(angle_1))))
    #angle_check_2 = np.arctan(abs((k - np.tan(angle_2)) / (1 + k * np.tan(angle_2))))

    point3_x = (-k * point1_x + point1_y) / (np.tan(angle_1) - k)
    point3_y = (-1 / k * point1_y + point1_x) / (1 / np.tan(angle_1) - 1 / k)
    depth = np.sqrt(point3_x ** 2 + point3_y ** 2)

    depth = np.where(np.isnan(depth), R_0, depth)
    angle_check_1 = np.where(np.isnan(angle_check_1), np.pi/2, angle_check_1)
    return depth, 1 / np.tan(angle_check_1) # sensitivity



# find possible nearest upper left lidar point from new phi and theta
def find_sample_from_angle(phi, theta, x_sample_deg, y_sample_deg, x_sample, upper_lim):
    # find possible nearest upper left lidar sample point
    if abs(phi / np.pi * 180) < x_sample_deg / 2:
        possible_x = x_sample // 2 - 1
    elif phi > 0:
        possible_x = x_sample // 2 - (phi / np.pi * 180 - x_sample_deg / 2) // x_sample_deg - 2
    else:
        possible_x = x_sample // 2 + (abs(phi / np.pi * 180) - x_sample_deg / 2) // x_sample_deg

    possible_y = (theta / np.pi * 180 - upper_lim) // y_sample_deg  # the vertical mapping start actually form upper_lim + y_sample_deg

    return int(possible_x), int(possible_y)


#def tri_area(x_1, y_1, x_2, y_2, x_3, y_3):
#    area = abs(x_1*y_2 + x_2*y_3 - x_3*y_2 - x_2*y_1)/2
#    return area


#def inside_polygon2(x, y, points):
#    """
#    Return True if a coordinate (x, y) is inside a polygon defined by
#    a list of verticies [(x1, y1), (x2, x2), ... , (xN, yN)].
#
#    Reference: http://www.ariel.com.au/a/python-point-int-poly.html
#    """
#    n = len(points)
#    inside = False
#    p1x, p1y = points[0]
#    for i in range(1, n + 1):
#        p2x, p2y = points[i % n]
#        if y > min(p1y, p2y):
#            if y <= max(p1y, p2y):
#                if x <= max(p1x, p2x):
#                    if p1y != p2y:
#                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
#                    if p1x == p2x or x <= xinters:
#                        inside = not inside
#        p1x, p1y = p2x, p2y
#    return inside


#def inside_polygon(x0,y0,x1,y1,x2,y2,x3,y3,x4,y4):
#    #print('-- check if inside')
#    inside = False
#    area_quad = abs(x1*y2 + x2*y3 + x3*y4 - x4*y3 - x3*y2 - x2*y1)/2
#    # https://en.wikipedia.org/wiki/Shoelace_formula
#    area_sum = tri_area(x0,y0,x1,y1,x2,y2) + tri_area(x0,y0,x2,y2,x3,y3) + tri_area(x0,y0,x3,y3,x4,y4) + tri_area(x0,y0,x1,y1,x4,y4)
#    if area_quad == area_sum:
#        inside = True
#    return inside
