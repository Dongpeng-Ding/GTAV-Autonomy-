
import cv2
import time
import math
import statistics
import numpy as np
import matplotlib.pyplot as plt
import itertools
import random

from Lidar_tools import find_lidar_phi_from_coord, find_lidar_theta_from_coord, find_lidar_theta_phi_from_image, \
    find_sample_from_angle, interpolation, polar_to_cartesian
from Lidar_tools_AG import coords_ro_move, build_roatation_matrix_3D




def build_matrix_from_line(target_line, option=0):

    try:
        angle = math.atan((target_line[1] - target_line[3]) / (target_line[0] - target_line[2]))
    except ZeroDivisionError:
        angle = np.pi / 2

    theta = np.pi / 2 - angle
    if option == 0:  # move to the position of the first point
        move_matrix = np.array([-target_line[2], -target_line[3]])
    else:  # move to the center of the line
        move_matrix = np.array([-(target_line[2] + target_line[0])/2, -(target_line[3] + target_line[1])/2])


    rotation_martix = np.array([[math.cos(theta), -math.sin(theta)],
                                [math.sin(theta), math.cos(theta)]])

    return move_matrix, rotation_martix, angle


def debug_c1(n,n_last, lines):  # for inherit, threshold must be the same with cluster_criterion
    lines_col1 = lines[n_last, 0:2]
    lines_col1 = np.expand_dims(lines_col1, axis=0)
    lines_col2 = lines[n_last, 2:4]
    lines_col2 = np.expand_dims(lines_col2, axis=0)
    line = lines[n]

    result = cluster_criterion(line, lines_col1, lines_col2)

    if result == 0:
        callback = '-- can\'t connect in the other dirction, correctly inherit the cluster'
    else:
        callback = '-- can connect in the other dirction, something wrong'

    return callback


#def rotation_to_line(n, lines_col1, lines_col2):  # rotate to the n line
#    try:
#        theta = math.atan((lines_col2[n, 1] - lines_col1[n, 1]) / (lines_col2[n, 0] - lines_col1[n, 0]))
#    except ZeroDivisionError:
#        theta = np.pi / 2
#    theta = np.pi / 2 - theta
#
#    move_matrix = np.array([-(lines_col1[n,0] + lines_col2[n,0])/2, -(lines_col1[n,1] + lines_col2[n,1])/2])
#    roatation_martix = np.array([[math.cos(theta), -math.sin(theta)],
#                                 [math.sin(theta), math.cos(theta)]])
#
#    lines_col1_new = np.dot((lines_col1 + move_matrix), np.linalg.inv(roatation_martix))
#    lines_col2_new = np.dot((lines_col2 + move_matrix), np.linalg.inv(roatation_martix))
#
#    return lines_col1_new, lines_col2_new


def cluster_criterion(n, lines_col1, lines_col2, threshold_angle=20, threshold_offset=1, threshold_dist=7):  # output a index vector, where == 1 means that line is connected with the target line
    threshold_angle = threshold_angle / 180 * np.pi
    sample_num = lines_col1.shape[0]

    move_matrix, roatation_martix, angle = build_matrix_from_line([lines_col1[n, 0], lines_col1[n, 1], lines_col2[n, 0], lines_col2[n, 1]], option=1)
    lines_col1_new = np.dot((lines_col1 + move_matrix), np.linalg.inv(roatation_martix))
    lines_col2_new = np.dot((lines_col2 + move_matrix), np.linalg.inv(roatation_martix))

    delta = lines_col2_new - lines_col1_new
    theta_new = np.arctan(delta[:, 1] / delta[:, 0])
    angle_diff = np.pi / 2 - abs(theta_new)
    x_diff = np.min(abs(np.hstack([lines_col1_new[:, 0].reshape(sample_num, 1), lines_col2_new[:, 0].reshape(sample_num, 1)])),axis=1)
    y_diff = np.min(abs(np.hstack([lines_col1_new[:, 1].reshape(sample_num, 1), lines_col2_new[:, 1].reshape(sample_num, 1)])),axis=1)
    connect_index_add = (angle_diff < threshold_angle) & (x_diff < threshold_offset) & (y_diff < threshold_dist)
    return connect_index_add

def cluster_criterion2(n, lines_col1, lines_col2, threshold_r=0.01):
    move_matrix, roatation_martix, angle = build_matrix_from_line([lines_col1[n, 0], lines_col1[n, 1], lines_col2[n, 0], lines_col2[n, 1]], option=1)
    lines_col1_new = np.dot((lines_col1 + move_matrix), np.linalg.inv(roatation_martix))
    lines_col2_new = np.dot((lines_col2 + move_matrix), np.linalg.inv(roatation_martix))

    delta = lines_col2_new - lines_col1_new
    k = delta[:, 1] / delta[:, 0]
    k = -1/k
    assert(not (np.any(k == np.inf) | np.any(k==0)))
    line_center = (lines_col2_new + lines_col1_new) / 2
    b = line_center[:,1] - k * line_center[:,0]
    intercept_x = np.abs(b/k)  # it's also the r1
    r2 = np.sqrt(intercept_x**2 + b**2)
    connect_index_add = np.abs(intercept_x - r2) < threshold_r
    return connect_index_add

def cluster_criterion3(n, lines_col1, lines_col2, threshold_angle=5, threshold_offset=0.5):
    threshold_angle = 180 - 2*threshold_angle
    threshold_angle = threshold_angle / 180 * np.pi
    sample_num = lines_col1.shape[0]

    move_matrix, roatation_martix, angle = build_matrix_from_line(
        [lines_col1[n, 0], lines_col1[n, 1], lines_col2[n, 0], lines_col2[n, 1]], option=1)
    lines_col1_new = np.dot((lines_col1 + move_matrix), np.linalg.inv(roatation_martix))
    lines_col2_new = np.dot((lines_col2 + move_matrix), np.linalg.inv(roatation_martix))

    delta = lines_col2_new - lines_col1_new
    k = delta[:, 1] / delta[:, 0]
    k2 = -1/k
    line_center = (lines_col2_new + lines_col1_new) / 2
    m = line_center[:,1] - k * line_center[:,0]
    m2 = line_center[:,1] - k2 * line_center[:,0]

    inter1 = np.vstack([np.zeros_like(m), m]).T
    inter2 = np.vstack([np.zeros_like(m2), m2]).T

    vector1 = inter1 - line_center
    length1 = np.sqrt(vector1[:,0]**2 + vector1[:,1]**2)
    base_vector1 = vector1[n,:]
    base_length1 = np.sqrt(vector1[n,0]**2 + vector1[n,1]**2)

    vector2 = inter2 - line_center
    length2 = np.sqrt(vector2[:, 0] ** 2 + vector2[:, 1] ** 2)
    base_vector2 = vector2[n, :]
    base_length2 = np.sqrt(vector2[n, 0] ** 2 + vector2[n, 1] ** 2)

    x_diff = np.min(abs(np.hstack([lines_col1_new[:, 0].reshape(sample_num, 1), lines_col2_new[:, 0].reshape(sample_num, 1)])),axis=1)

    angle1 = np.arccos(np.dot(vector1, base_vector1) / length1 / base_length1)
    angle2 = np.arccos(np.dot(vector2, base_vector2) / length2 / base_length2)

    connect_index_add1 = angle1 - angle2 > threshold_angle
    connect_index_add1[np.where(k == math.inf)] = 1
    connect_index_add1[np.where(k == -math.inf)] = 1
    connect_index_add2 = x_diff < threshold_offset
    connect_index_add2[np.where(k != math.inf)] = 1
    connect_index_add2[np.where(k != -math.inf)] = 1

    connect_index_add = connect_index_add1 & connect_index_add2
    #print(connect_index_add)

    return connect_index_add


def iter_core(n_current, n_last, lines, check_state, connect_index_initial, cluster_index, inherit_C):
    lines_col1 = lines[:, 0:2]
    lines_col2 = lines[:, 2:4]
    #sample_num = lines.shape[0]
    #line = lines[n_current]

    if check_state[n_current] == 0: # this line is not checked before
        check_state[n_current] = 1

        connect_index_add = cluster_criterion(n_current, lines_col1, lines_col2)
        connect_index_current = connect_index_initial | connect_index_add

        order = 'continue'
        n_next = 0
        for x in connect_index_add:
            if x == 1:  # recursive
                order_last, C, check_state_deeper, connect_index_deeper = iter_core(n_next, n_current, lines, check_state, connect_index_current, cluster_index, inherit_C)
                connect_index_current = connect_index_deeper | connect_index_current
                check_state = check_state_deeper | check_state
                if order_last == 'inherit C and continue':
                    inherit_C = C
                    order = 'inherit C and continue'
            n_next += 1

        return order, inherit_C, check_state, connect_index_current

    elif check_state[n_current] == 1: # this line is checked before, but not assigned with a cluster number
        # actually do nothing here
        C = None  # not used
        order = 'continue'
        return order, C, check_state, connect_index_initial

    elif check_state[n_current] == 2: # this line is checked before, and assigned with a cluster number
        # this part is useful if the criterion let line A connected to B, but cant let line B connect to A

        #print('inherit from old cluster!')
        #print(debug_c1(n_current, n_last, lines))

        C = cluster_index[n_current]
        # note: if the current process connects with more than 1 old cluster,
        # only the last C (cluster num) will be inherited, all previous C will be overwrite.
        # But connections from all old clusters will be added from the line below.

        connect_index_initial[np.where(cluster_index == C)] = 1  # connection from old cluster is added
        order = 'inherit C and continue'
        return order, C, check_state, connect_index_initial


def connect_lines(lines):

    lines_col1 = lines[:, 0:2]
    lines_col2 = lines[:, 2:4]
    sample_num = lines.shape[0]
    cluster_index = np.zeros(sample_num, dtype=int)
    check_state = np.zeros(sample_num, dtype=int)

    n_current = 0
    for line in lines:
        connect_index_initial = np.zeros(sample_num, dtype=bool)
        connect_index_initial[n_current] = 1
        inherit_C = 0

        # the first few lines are same as in iter_core
        if check_state[n_current] == 0:
            check_state[n_current] = 1

            connect_index_add = cluster_criterion(n_current, lines_col1, lines_col2)
            connect_index_current = connect_index_initial | connect_index_add

            order = 'set new cluster'
            n_next = 0
            for x in connect_index_add:
                if x == 1:
                    order_last, C, check_state_deeper, connect_index_deeper = iter_core(n_next,n_current, lines, check_state, connect_index_current, cluster_index, inherit_C)
                    connect_index_current = connect_index_deeper | connect_index_current
                    check_state = check_state_deeper | check_state
                    if order_last == 'inherit C and continue':
                        inherit_C = C
                        order = 'add to previous cluster'
                n_next += 1

            #### DEBUG
            if np.any(cluster_index[np.where(connect_index_current == 1)] > 0):
                print('warning: cluster override!!!!!!!!!!!!!!!!!!!!!!!!!!!!!')
                overrided_cluster = cluster_index[np.where(connect_index_current == 1)]
                if order == 'add to previous cluster':
                    if np.any(overrided_cluster == inherit_C):
                        print('-- cluster expand')
                    if np.any((overrided_cluster != inherit_C) & (overrided_cluster != 0)):
                        print('-- cluster rewrite')
                else:
                    print('-- override but no inherit detected!')
            ##

            if order == 'add to previous cluster':
                cluster_index[np.where(connect_index_current == 1)] = inherit_C
                check_state[np.where(connect_index_current == 1)] = 2

            else:  # means: not connect to previous cluster
                C = np.max(cluster_index) + 1  # set to a new cluster
                cluster_index[np.where(connect_index_current == 1)] = C
                check_state[np.where(connect_index_current == 1)] = 2

                #### DEBUG
                if np.any(check_state ==1):
                    assert(0)
                ##

        #### DEBUG
        else:
            if check_state[n_current] == 1:
                assert(cluster_index[n_current] != 0)
            elif check_state[n_current] == 2:
                assert (cluster_index[n_current] != 0)
        ##

        n_current += 1

    #### DEBUG
    assert (not np.any(cluster_index == 0))
    ##

    normalized_cluster_index = np.zeros_like(cluster_index)
    maximum = np.max(cluster_index)
    num = 1
    for x in range(maximum+1):
        if np.any(cluster_index == x):
            normalized_cluster_index[np.where(cluster_index == x)] = num
            num += 1
    #print(cluster_index)
    #print('normalized_cluster_index: ',normalized_cluster_index)
    return normalized_cluster_index


def cluster_split_fit(points, points_new, threshold, max_loop=20):  # fit the cluster with 2 lines
    rms_error_last = math.inf
    best_by_now_l, best_by_now_r = None, None
    best_by_now_l_length, best_by_now_r_length = None, None

    for x in range(max_loop):
        if x == max_loop - 1:
            if best_by_now_l is not None and best_by_now_r is not None:
                print('split done, but not optimized!!!!!!!!!!!!!!!!!!!!!!!!!')
                return best_by_now_l, best_by_now_l_length, best_by_now_r, best_by_now_r_length, 'split done'
            else:
                print('cant split cluster!!!!!!!!!!!!!!!!!!!!!!!!!!')
                return [0,0,0,0], [0,0], [0,0,0,0], [0,0], 'cant split'

        group_left = points[np.where(points_new[:,0] < 0), :]
        group_right = points[np.where(points_new[:,0] > 0), :]
        group_left, group_right = group_left[0], group_right[0]

        [vx_l, vy_l, x0_l, y0_l] = cv2.fitLine(group_left, distType=cv2.cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        [vx_r, vy_r, x0_r, y0_r] = cv2.fitLine(group_right, distType=cv2.cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        x0_l, y0_l, x0_r, y0_r = x0_l[0], y0_l[0], x0_r[0], y0_r[0]

        try:
            theta_l = math.atan(vy_l / vx_l)
        except ZeroDivisionError:
            theta_l = np.pi / 2
        theta_l = np.pi / 2 - theta_l
        move_matrix_l = np.array([-x0_l, -y0_l])
        roatation_martix_l = np.array([[math.cos(theta_l), -math.sin(theta_l)],
                                     [math.sin(theta_l), math.cos(theta_l)]])

        group_left_new = np.dot((group_left + move_matrix_l), np.linalg.inv(roatation_martix_l))
        rms_error_l = np.sqrt(np.sum(group_left_new[:, 0] ** 2) / group_left_new.shape[0])
        approxi_length_l = [np.max(group_left_new[:, 1]), np.min(group_left_new[:, 1])]

        try:
            theta_r = math.atan(vy_r / vx_r)
        except ZeroDivisionError:
            theta_r = np.pi / 2
        theta_r = np.pi / 2 - theta_r
        move_matrix_r = np.array([-x0_r, -y0_r])
        roatation_martix_r = np.array([[math.cos(theta_r), -math.sin(theta_r)],
                                      [math.sin(theta_r), math.cos(theta_r)]])

        group_right_new = np.dot((group_right + move_matrix_r), np.linalg.inv(roatation_martix_r))
        rms_error_r = np.sqrt(np.sum(group_right_new[:, 0] ** 2) / group_right_new.shape[0])
        approxi_length_r = [np.max(group_right_new[:, 1]), np.min(group_right_new[:, 1])]

        # check rmse
        rms_error_new = (rms_error_l + rms_error_r) / 2

        if rms_error_new < threshold:
            print('split takes: ', x+1, ' loop')
            return [vx_l, vy_l, x0_l, y0_l], approxi_length_l, [vx_r, vy_r, x0_r, y0_r], approxi_length_r, 'split done'

        else:  # split the cluster again
            try:
                theta = math.atan((vy_l + vy_r) / (vx_l + vx_r))
            except ZeroDivisionError:
                theta = np.pi / 2
            theta = np.pi / 2 - theta
            move_matrix = np.array([-(x0_l + x0_r) / 2, -(y0_l + y0_r) / 2])
            roatation_martix = np.array([[math.cos(theta), -math.sin(theta)],
                                         [math.sin(theta), math.cos(theta)]])

            points_new = np.dot((points + move_matrix), np.linalg.inv(roatation_martix))

        if rms_error_new < rms_error_last:
            best_by_now_l, best_by_now_r = [vx_l, vy_l, x0_l, y0_l], [vx_r, vy_r, x0_r, y0_r]
            best_by_now_l_length, best_by_now_r_length = approxi_length_l, approxi_length_r

        rms_error_last = rms_error_new


def cluster_fit(lines, cluster_index, rms_threshold=0.8):  # fit a line for each cluster
    cluster_index_2 = []
    #cluster_fit_error = []
    line_length = []
    new_lines = np.empty([0,4])
    for x in range(1, np.max(cluster_index)+1):
        line = lines[np.where(cluster_index == x), :][0]
        line_col1 = line[:, 0:2]
        line_col2 = line[:, 2:4]
        points = np.vstack([line_col1,line_col2])
        if points.size ==0:
            print(cluster_index)
            print(x)
            print(lines)
            assert(0)
        [vx, vy, x0, y0] = cv2.fitLine(points, distType=cv2.cv2.DIST_L2, param=0, reps=0.01, aeps=0.01)
        x0, y0 = x0[0], y0[0]

        # rotate to the fitted line
        try:
            angle = math.atan(vy / vx)
        except ZeroDivisionError:
            angle = np.pi / 2

        theta = np.pi / 2 - angle
        move_matrix = np.array([-x0, -y0])
        roatation_martix = np.array([[math.cos(theta), -math.sin(theta)],
                                     [math.sin(theta), math.cos(theta)]])

        points_new = np.dot((points + move_matrix), np.linalg.inv(roatation_martix))

        # calculate error and line length
        rms_error = np.sqrt(np.sum(points_new[:, 0]**2) / points_new.shape[0])
        length_max = np.max(points_new[:, 1])
        length_min = np.min(points_new[:, 1])
        #cluster_fit_error.append(rms_error)


        if rms_error > rms_threshold:
            # try to fit the cluster with 2 lines
            [vx_l, vy_l, x0_l, y0_l], [length_max_l, length_min_l], [vx_r, vy_r, x0_r, y0_r], [length_max_r, length_min_r], answer \
                = cluster_split_fit(points, points_new, rms_threshold, max_loop=10)
            if answer == 'split done':
                try:
                    angle_l = math.atan(vy_l / vx_l)
                except ZeroDivisionError:
                    angle_l = np.pi / 2
                try:
                    angle_r = math.atan(vy_r / vx_r)
                except ZeroDivisionError:
                    angle_r = np.pi / 2

                x1_l = x0_l + length_max_l * math.cos(angle_l)
                y1_l = y0_l + length_max_l * math.sin(angle_l)
                x2_l = x0_l + length_min_l * math.cos(angle_l)
                y2_l = y0_l + length_min_l * math.sin(angle_l)

                x1_r = x0_r + length_max_r * math.cos(angle_r)
                y1_r = y0_r + length_max_r * math.sin(angle_r)
                x2_r = x0_r + length_min_r * math.cos(angle_r)
                y2_r = y0_r + length_min_r * math.sin(angle_r)

                new_line = np.array([[x1_l, y1_l, x2_l, y2_l], [x1_r, y1_r, x2_r, y2_r]])
                new_lines = np.concatenate((new_lines, new_line), axis=0)
                cluster_index_2.extend([x, x])
                line_length.extend([length_max_l - length_min_l, length_max_r - length_min_r])
            else:
                x1 = x0 + length_max * math.cos(angle)
                y1 = y0 + length_max * math.sin(angle)
                x2 = x0 + length_min * math.cos(angle)
                y2 = y0 + length_min * math.sin(angle)

                new_line = np.array([[x1, y1, x2, y2]])
                new_lines = np.concatenate((new_lines, new_line), axis=0)
                cluster_index_2.extend([x])
                line_length.extend([length_max - length_min])
        else:
            x1 = x0 + length_max * math.cos(angle)
            y1 = y0 + length_max * math.sin(angle)
            x2 = x0 + length_min * math.cos(angle)
            y2 = y0 + length_min * math.sin(angle)

            new_line = np.array([[x1, y1, x2, y2]])
            new_lines = np.concatenate((new_lines, new_line), axis=0)
            cluster_index_2.extend([x])
            line_length.extend([length_max - length_min])


    return new_lines, cluster_index_2, line_length



def search_lane(target_line, lines, c1, cluster_index_2):  # for one line, search the nearest line, ie. forms a lane

    target_line_x1, target_line_y1, target_line_x2, target_line_y2 = target_line[0], target_line[1], target_line[2], target_line[3]

    length_max = None
    length_min = None
    center_x = None
    center_y = None
    lane_angle = None
    saved_angle_diff = None
    saved_road_width = math.inf
    saved_c2 = None
    flag_invalid = None

    count2 = 0
    for other_line in lines:
        c2 = cluster_index_2[count2]
        if c1 == c2:
            continue

        other_line_x1 = other_line[0]
        other_line_y1 = other_line[1]
        other_line_x2 = other_line[2]
        other_line_y2 = other_line[3]

        line_x_center = (target_line_x1 + target_line_x2) / 2
        line_y_center = (target_line_y1 + target_line_y2) / 2
        other_line_x_center = (other_line_x1 + other_line_x2) / 2
        other_line_y_center = (other_line_y1 + other_line_y2) / 2

        delta_x = target_line_x1 - target_line_x2
        delta_other_x = other_line_x1 - other_line_x2
        delta_y = target_line_y1 - target_line_y2
        delta_other_y = other_line_y1 - other_line_y2

        p = delta_other_x * delta_y - delta_x * delta_other_y  # i.e slope 1 == slope 2
        p2 = - p

        if p != 0:  # not parallel
            # intercept point of the two lines
            y0 = ((target_line_x1 - other_line_x1) * delta_y * delta_other_y + other_line_y1 * delta_other_x * delta_y - target_line_y1 * delta_x * delta_other_y) / p
            x0 = ((target_line_y1 - other_line_y1) * delta_x * delta_other_x + other_line_x1 * delta_other_y * delta_x - target_line_x1 * delta_y * delta_other_x) / p2

            # vector from the intercept point to line center
            vector1_x = x0 - line_x_center
            vector1_y = y0 - line_y_center
            vector2_x = x0 - other_line_x_center
            vector2_y = y0 - other_line_y_center
            vector1_length = np.sqrt(vector1_x ** 2 + vector1_y ** 2)
            vector2_length = np.sqrt(vector2_x ** 2 + vector2_y ** 2)

            # angle difference from the two lines
            try:
                angle_diff = math.acos((vector1_x * vector2_x + vector1_y * vector2_y) / (vector1_length * vector2_length))
            except ValueError:
                print((vector1_x * vector2_x + vector1_y * vector2_y) / (vector1_length * vector2_length))
                assert(0)
                #angle_diff = 0

            # the center point of the additional line, that connects the center points of the two lines
            # i.e it is the lane center point
            x00 = line_x_center * vector2_length / (vector1_length + vector2_length) + \
                  other_line_x_center * vector1_length / (vector1_length + vector2_length)
            y00 = line_y_center * vector2_length / (vector1_length + vector2_length) + \
                  other_line_y_center * vector1_length / (vector1_length + vector2_length)

            # rotate to the direction of the vector: intercept point of lane lines -- lane center point
            move_matrix, rotation_martix, angle = build_matrix_from_line([x0, y0, x00, y00])

            line2roll = np.array([[target_line_x1, target_line_y1], [target_line_x2, target_line_y2]])
            other_line2roll = np.array([[other_line_x1, other_line_y1], [other_line_x2, other_line_y2]])

            line_new = np.dot((line2roll + move_matrix), np.linalg.inv(rotation_martix))
            other_line_new = np.dot((other_line2roll + move_matrix), np.linalg.inv(rotation_martix))

            # check vertical offset
            # both should < 0 or < threshold
            offset1 = np.min(line_new[:, 1]) - np.max(other_line_new[:, 1])
            offset2 = np.min(other_line_new[:, 1]) - np.max(line_new[:, 1])

            if offset1 >0 or offset2 > 0:  # the two lane lines are not opposite to each other
                continue

            # calculate lane width, lane width is thelength of the additional line mentioned above
            delta = line_new[0, :] - line_new[1, :]
            k = delta[1] / delta[0]
            intercept_1 = line_new[0, 0] - line_new[0, 1] / k

            delta = other_line_new[0, :] - other_line_new[1, :]
            k = delta[1] / delta[0]
            intercept_2 = other_line_new[0, 0] - other_line_new[0, 1] / k

            road_width = abs(intercept_1 - intercept_2)

            # if width is the smallest, further calculate the lane length
            if road_width < saved_road_width:
                # each pair of the 4 points from both lines forms a line, calculate theirs intercept to y axis
                delta = line_new[0, :] - other_line_new[0, :]
                k = delta[1] / delta[0]
                length1 = line_new[0, 1] - line_new[0, 0] * k

                delta = line_new[1, :] - other_line_new[0, :]
                k = delta[1] / delta[0]
                length2 = line_new[1, 1] - line_new[1, 0] * k

                delta = line_new[1, :] - other_line_new[1, :]
                k = delta[1] / delta[0]
                length3 = line_new[1, 1] - line_new[1, 0] * k

                delta = line_new[0, :] - other_line_new[1, :]
                k = delta[1] / delta[0]
                length4 = line_new[0, 1] - line_new[0, 0] * k

                # the max and min of the intercept are the upper and lower point of the lane
                length_max = max([length1, length2, length3, length4])
                length_min = min([length1, length2, length3, length4])
                center_x = x00
                center_y = y00
                lane_angle = angle
                saved_c2 = c2
                saved_angle_diff = angle_diff
                saved_road_width = road_width

                # the intercept point x0y0 should not on one of the line, if true raise a flag
                if line_new[0,0] * line_new[1,0] < 0 or other_line_new[0,0] * other_line_new[1,0] < 0:
                    flag_invalid = 1
                else:
                    flag_invalid = 0
                #flag_invalid = 0

        else:  # parallel lines

            # rotate to one of the line
            move_matrix, rotation_martix, angle = build_matrix_from_line(target_line)
            #try:
            #    angle = math.atan((target_line_y2 - target_line_y1) / (target_line_x2 - target_line_x1))
            #except ZeroDivisionError:
            #    angle = np.pi / 2
            #theta = np.pi / 2 - angle
#
            x00 = (line_x_center + other_line_x_center) / 2
            y00 = (line_y_center + other_line_y_center) / 2
#
            #move_matrix = np.array([-x00, -y00])
            #rotation_martix = np.array([[math.cos(theta), -math.sin(theta)],
            #                             [math.sin(theta), math.cos(theta)]])

            line2roll = np.array([[target_line_x1, target_line_y1], [target_line_x2, target_line_y2]])
            other_line2roll = np.array([[other_line_x1, other_line_y1], [other_line_x2, other_line_y2]])

            line_new = np.dot((line2roll + move_matrix), np.linalg.inv(rotation_martix))
            other_line_new = np.dot((other_line2roll + move_matrix), np.linalg.inv(rotation_martix))

            # check vertical offset
            # both should < 0 or < threshold
            offset1 = np.min(line_new[:, 1]) - np.max(other_line_new[:, 1])
            offset2 = np.min(other_line_new[:, 1]) - np.max(line_new[:, 1])
            if offset1 > 0 or offset2 > 0:
                continue

            road_width = abs(other_line_new[0, 0]) * 2
            if road_width < saved_road_width:
                length1 = (line_new[0, 1] + other_line_new[0, 1]) / 2
                length2 = (line_new[1, 1] + other_line_new[0, 1]) / 2
                length3 = (line_new[1, 1] + other_line_new[1, 1]) / 2
                length4 = (line_new[0, 1] + other_line_new[1, 1]) / 2

                length_max = max([length1, length2, length3, length4])
                length_min = min([length1, length2, length3, length4])
                center_x = x00
                center_y = y00
                lane_angle = angle
                saved_c2 = c2
                saved_angle_diff = 0
                saved_road_width = road_width

                flag_invalid = 0

        count2 += 1

    return saved_c2, saved_angle_diff, saved_road_width, length_max, length_min, center_x, center_y, lane_angle, flag_invalid


def esti_lane(lines, cluster_index, max_angle=20, min_road_width=3., max_road_width=10):  # for each line, check if its R and L lane exist, and build them
    max_angle = max_angle / 180 * np.pi
    lines_col1 = lines[:, 0:2]
    lines_col2 = lines[:, 2:4]
    cluster_num = np.max(cluster_index)
    cluster_index = np.array(cluster_index)
    checked_index = np.zeros([cluster_num, cluster_num], dtype=bool)
    lanes = np.empty([0,4])

    for c1 in range(1, cluster_num + 1):  # loop for all clusters
        target_line = lines[np.where(cluster_index == c1), :][0]

        c2_r = None  #cluster number for left lane line
        c2_l = None  #cluster number for right lane line
        if target_line.shape[0] == 1:  # the cluster contains only one lane line

            # rotate to the current target_line in this loop
            target_line = target_line[0]
            move_matrix, rotation_martix, ang = build_matrix_from_line(target_line)
            lines_col1_new = np.dot((lines_col1 + move_matrix), np.linalg.inv(rotation_martix))
            lines_col2_new = np.dot((lines_col2 + move_matrix), np.linalg.inv(rotation_martix))

            # group left lines and right lines
            index_left = (lines_col1_new[:, 0] < 0) & (lines_col2_new[:, 0] < 0)
            index_right = (lines_col1_new[:, 0] > 0) & (lines_col2_new[:, 0] > 0)

            left_lines = lines[index_left, :]
            left_cluster_index = cluster_index[index_left]
            right_lines = lines[index_right, :]
            right_cluster_index = cluster_index[index_right]

            # search left and right lines to find a lane
            c2_l, angle_diff_l, road_width_l, length_max_l, length_min_l, center_x_l, center_y_l, lane_angle_l, flag_invalid_l = \
                search_lane(target_line, left_lines, c1, left_cluster_index)

            c2_r, angle_diff_r, road_width_r, length_max_r, length_min_r, center_x_r, center_y_r, lane_angle_r, flag_invalid_r = \
                search_lane(target_line, right_lines, c1, right_cluster_index)

            if c2_l is not None:  # there is a candidate for left lane
                if checked_index[c1 - 1, c2_l - 1] == 0 or checked_index[c2_l - 1, c1 - 1] == 0:
                    if angle_diff_l < max_angle and min_road_width < road_width_l < max_road_width and flag_invalid_l == 0:
                        x1 = center_x_l + length_max_l * math.cos(lane_angle_l)
                        y1 = center_y_l + length_max_l * math.sin(lane_angle_l)
                        x2 = center_x_l + length_min_l * math.cos(lane_angle_l)
                        y2 = center_y_l + length_min_l * math.sin(lane_angle_l)
                        lanes = np.concatenate((lanes, [[x1, y1, x2, y2]]), axis=0)
                        # lane_boundary.append([c1,c2])
                        checked_index[c1 - 1, c2_l - 1], checked_index[c2_l - 1, c1 - 1] = 1, 1

            if c2_r is not None:  # there is a candidate for right lane
                if checked_index[c1 - 1, c2_r - 1] == 0 or checked_index[c2_r - 1, c1 - 1] == 0:
                    if angle_diff_r < max_angle and min_road_width < road_width_r < max_road_width and flag_invalid_r == 0:
                        x1 = center_x_r + length_max_r * math.cos(lane_angle_r)
                        y1 = center_y_r + length_max_r * math.sin(lane_angle_r)
                        x2 = center_x_r + length_min_r * math.cos(lane_angle_r)
                        y2 = center_y_r + length_min_r * math.sin(lane_angle_r)
                        lanes = np.concatenate((lanes, [[x1, y1, x2, y2]]), axis=0)
                        # lane_boundary.append([c1,c2])
                        checked_index[c1 - 1, c2_r - 1], checked_index[c2_r - 1, c1 - 1] = 1, 1

        else:  # it's splitted cluster, the cluster contains 2 lane lines
            assert (target_line.shape[0] == 2)

            # check for each lane lines
            for n in range(2):
                if n == 0:
                    one_line = target_line[0, :]
                    other_line = target_line[1, :]
                else:
                    one_line = target_line[1, :]
                    other_line = target_line[0, :]

                # rotation to one of the two cluster
                move_matrix, rotation_martix, ang = build_matrix_from_line(one_line)

                line2_center = np.array([(other_line[0] + other_line[2]) / 2, (other_line[1] + other_line[3]) / 2])
                line2_center_new = np.dot((line2_center + move_matrix), np.linalg.inv(rotation_martix))

                lines_col1_new = np.dot((lines_col1 + move_matrix), np.linalg.inv(rotation_martix))
                lines_col2_new = np.dot((lines_col2 + move_matrix), np.linalg.inv(rotation_martix))

                if line2_center_new[0] < 0:  # the other line is on the left. only check right side

                    index_right = (lines_col1_new[:, 0] > 0) & (lines_col2_new[:, 0] > 0)
                    right_lines = lines[index_right, :]
                    right_cluster_index = cluster_index[index_right]
                    c2_r, angle_diff_r, road_width_r, length_max_r, length_min_r, center_x_r, center_y_r, lane_angle_r, flag_invalid_r = \
                        search_lane(one_line, right_lines, c1, right_cluster_index)

                    if c2_r is not None:
                        if checked_index[c1 - 1, c2_r - 1] == 0 or checked_index[c2_r - 1, c1 - 1] == 0:
                            if angle_diff_r < max_angle and min_road_width < road_width_r < max_road_width and flag_invalid_r == 0:
                                x1 = center_x_r + length_max_r * math.cos(lane_angle_r)
                                y1 = center_y_r + length_max_r * math.sin(lane_angle_r)
                                x2 = center_x_r + length_min_r * math.cos(lane_angle_r)
                                y2 = center_y_r + length_min_r * math.sin(lane_angle_r)
                                lanes = np.concatenate((lanes, [[x1, y1, x2, y2]]), axis=0)
                                # lane_boundary.append([c1,c2])
                                checked_index[c1 - 1, c2_r - 1], checked_index[c2_r - 1, c1 - 1] = 1, 1

                else:

                    index_left = (lines_col1_new[:, 0] < 0) & (lines_col2_new[:, 0] < 0)
                    left_lines = lines[index_left, :]
                    left_cluster_index = cluster_index[index_left]
                    c2_l, angle_diff_l, road_width_l, length_max_l, length_min_l, center_x_l, center_y_l, lane_angle_l, flag_invalid_l = \
                        search_lane(one_line, left_lines, c1, left_cluster_index)

                    if c2_l is not None:
                        if checked_index[c1 - 1, c2_l - 1] == 0 or checked_index[c2_l - 1, c1 - 1] == 0:
                            if angle_diff_l < max_angle and min_road_width < road_width_l < max_road_width and flag_invalid_l == 0:
                                x1 = center_x_l + length_max_l * math.cos(lane_angle_l)
                                y1 = center_y_l + length_max_l * math.sin(lane_angle_l)
                                x2 = center_x_l + length_min_l * math.cos(lane_angle_l)
                                y2 = center_y_l + length_min_l * math.sin(lane_angle_l)
                                lanes = np.concatenate((lanes, [[x1, y1, x2, y2]]), axis=0)
                                # lane_boundary.append([c1,c2])
                                checked_index[c1 - 1, c2_l - 1], checked_index[c2_l - 1, c1 - 1] = 1, 1
    return lanes


def line_compare(line1, line2, theta_thresh, rho_thresh): # compare lines in theta rho space
    x_trans = (line1[0] + line1[2]) / 2
    y_trans = (line1[1] + line1[3]) / 2

    line1 = line1 - np.array([x_trans, y_trans, x_trans, y_trans])
    line2 = line2 - np.array([x_trans, y_trans, x_trans, y_trans])

    # change to parameter space, and compare
    try:
        theta1 = math.atan((line1[0] - line1[2])/(line1[3] - line1[1]))
    except ZeroDivisionError:
        theta1 = np.pi/2
    rho1 = line1[0] * math.cos(theta1) + line1[1] * math.sin(theta1)

    try:
        theta2 = math.atan((line2[0] - line2[2])/(line2[3] - line2[1]))
    except ZeroDivisionError:
        theta2 = np.pi/2
    rho2 = line2[0] * math.cos(theta2) + line2[1] * math.sin(theta2)

    if abs(rho1 - rho2) < rho_thresh and abs(theta1 - theta2) < theta_thresh:
        return True
    else:
        return False


def line_tracker2(lines_now, lines_last, delta_yaw, move_x, move_z):

    tracked_index = np.zeros([lines_now.shape[0],], dtype=bool)  # ######################### should update to use simple list
    lines_now_col1 = lines_now[:, 0:2]
    lines_now_col2 = lines_now[:, 2:4]

    move_matrix = np.array([move_x, move_z])
    roatation_martix = np.array([[math.cos(-delta_yaw), -math.sin(-delta_yaw)],
                                   [math.sin(-delta_yaw), math.cos(-delta_yaw)]])

    lines_last_col1 = np.dot(lines_now_col1, np.linalg.inv(roatation_martix)) + move_matrix
    lines_last_col2 = np.dot(lines_now_col2, np.linalg.inv(roatation_martix)) + move_matrix

    lines_last_esti = np.hstack((lines_last_col1, lines_last_col2))

    n = 0
    for line_last_esti in lines_last_esti.astype(int):
        for line_last in lines_last.astype(int):
            tracked = line_compare(line_last, line_last_esti, theta_thresh=np.pi/180*5, rho_thresh=0.5)
            if tracked == 1:
                tracked_index[n] = 1

        n += 1

    return tracked_index, lines_last_esti


class lane_finder:
    def __init__(self, Lidar_info, Camera_info2):
        #self.initialized = 0
        self.reference_x = None
        self.reference_y = None
        self.marker_ground = None
        self.depth_image = None
        self.image = None
        self.dt_in_game = None
        self.delta_yaw = None
        self.roll = None
        self.pitch = None
        self.speed_now = None
        self.acc = None
        self.lanes_last = None
        self.lines_last = None
        self.warp_matrix = None
        self.image_warped = None
        self.mask_warped = None
        self.focus = Camera_info2['focus']
        self.v_sample = Camera_info2['v_sample']
        self.h_sample = Camera_info2['h_sample']
        self.x_sample = Lidar_info['x_sample']
        self.x_sample_deg = Lidar_info['x_sample_deg']
        self.y_sample = Lidar_info['y_sample']
        self.y_sample_deg = Lidar_info['y_sample_deg']
        self.upper_lim = Lidar_info['upper_lim']

        self.move_z = None
        self.move_x = None

        self.mask = None
        self.mean_r = None
        self.mean_g = None
        self.mean_b = None


    def input(self, reference_x, reference_y, marker_ground, depth_image, image, delta_yaw, roll, pitch, acc, speed_now, dt_in_game):
        self.reference_x = reference_x
        self.reference_y = reference_y
        self.marker_ground = marker_ground
        self.depth_image = depth_image
        self.image = image
        self.delta_yaw = delta_yaw
        self.roll = roll
        self.pitch = pitch
        self.speed_now = speed_now
        self.acc = acc
        self.dt_in_game = dt_in_game
        self.warp_matrix = None


    def calculate_movement(self):
        dt_in_game = self.dt_in_game
        acc = self.acc
        speed_last = self.speed_now - acc * dt_in_game

        if self.delta_yaw != 0:
            #  simple ver. only using the yaw to estimate, in world coord: x,y,z refer to left, up, front
            horizontal_turn_radius = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) / self.delta_yaw
            self.move_z = horizontal_turn_radius * math.sin(self.delta_yaw)
            self.move_x = horizontal_turn_radius * (1 - math.cos(self.delta_yaw))

        else:
            self.move_z = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2)
            self.move_x = 0


    def roi(self):
        reference_x = self.reference_x
        reference_y = self.reference_y
        marker_ground = self.marker_ground
        image = self.image

        # check all vertices of the ground mask, except the two lowest vertices left and right
        vertices = []
        for x in range(marker_ground.shape[1]):
            for y in range(marker_ground.shape[0]):
                if marker_ground[y, x] == 1:
                    vertices.append([reference_x[y, x], reference_y[y, x]])
                    break

        vertices = np.array(vertices)

        # add the two lowest vertices left and right to the image bottom
        # (the lidar has smaller sight than the camera)
        right_end = np.max(vertices[:, 0])
        left_end = np.min(vertices[:, 0])
        vertices = np.append(vertices, [[right_end, image.shape[0]], [left_end, image.shape[0]]], axis=0)


        # build a mask of region of interest, and pre mask the image to calculate means in each channel
        mask = np.zeros([image.shape[0], image.shape[1]], dtype='uint8')
        cv2.fillPoly(mask, [vertices.astype('int32')], 255)
        self.mask = mask

        image_r = image[..., 0].astype(float)
        image_r[np.where(mask == 0)] = np.nan
        image_g = image[..., 0].astype(float)
        image_g[np.where(mask == 0)] = np.nan
        image_b = image[..., 0].astype(float)
        image_b[np.where(mask == 0)] = np.nan

        # mean of the rest pixels in each channel. this will be used to adjust images later
        self.mean_r = np.nanmean(image[..., 0])
        self.mean_g = np.nanmean(image[..., 1])
        self.mean_b = np.nanmean(image[..., 2])


    def rotation_correction(self): # eliminate vehicle's roll from the image
        image = self.image
        roll = self.roll

        height = image.shape[0]
        width = image.shape[1]

        roatation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), - roll, 1)
        self.image_roll_correct = cv2.warpAffine(image, roatation_matrix, (width, height))
        self.mask_roll_correct = cv2.warpAffine(self.mask, roatation_matrix, (width, height))


    def warp(self): # warp images to bird'S eyes view
        reference_x = self.reference_x
        reference_y = self.reference_y
        marker_ground = self.marker_ground
        depth_image = self.depth_image
        image = self.image_roll_correct
        mask = self.mask_roll_correct
        image_height = self.v_sample
        image_width = self.h_sample

        half_image_width = image_width // 2
        start_base_width = 200 # width of the bottom line from the vertices which build transformation matrix
        k = []  # slope of the side line from the vertices which build transformation matrix

        # calculate k by using lidar samples with a width from left x to right x, height up to max_distance
        size = 20
        x_r = marker_ground.shape[1] // 2 + size + 1
        x_l = x_r - 1 - size
        max_distance = 35  # m

        #delta = 0
        distance = 0
        pixel_pos = math.inf
        for x in range(x_l, x_r):

            base_width = start_base_width
            distance_temp = 0
            pixel_pos_temp = 0
            for y in range(marker_ground.shape[0] - 2, -1, -1):  # calculate from bottom to up
                if depth_image[y, x] >= max_distance or marker_ground[y, x] == 0:
                    break
                else:
                    v_diff = reference_y[y + 1, x] - reference_y[y, x]  # vertical pixel diff
                    decrease_ratio = depth_image[y + 1, x] / depth_image[y, x]  # road width decrease ratio
                    d_diff = base_width * (1 - decrease_ratio)
                    k.append(d_diff / v_diff)

                    base_width = base_width * decrease_ratio
                    #delta_temp += v_diff
                    pixel_pos_temp = reference_y[y, x]
                    distance_temp = depth_image[y, x]

            if distance_temp > distance and pixel_pos > pixel_pos_temp:
                distance = distance_temp  # update farthest distance
                pixel_pos = pixel_pos_temp  # update highest pixel position in image

        if len(k)>0:
            k = statistics.mean(k)
        else:
            k = 1.8
            pixel_pos = 179
            distance = max_distance

        ## warp:
        # set the size of new image after warp 450 * 600
        half_new_width = 225
        new_height = 600

        delta2 = image_height - reference_y[-1, x_r] - 1  # distance from center lowest lider point to image bottom
        start_base_width = start_base_width + k * delta2  # correct the bottom line

        left_base = half_image_width - start_base_width  # pixel position of this bottom line
        right_base = half_image_width + start_base_width

        left_base_2 = int(half_new_width - start_base_width/11)  # pixel position after warp
        right_base_2 = int(half_new_width + start_base_width/11)

        # set up vertices before warp
        pts1 = np.float32([[left_base + k * (image_height - pixel_pos), pixel_pos], [right_base - k * (image_height - pixel_pos), pixel_pos],
                           [left_base, image_height - 1], [right_base, image_height - 1]])

        # calculate the vertical position of highest vertices after warp
        if distance < max_distance:
            position = int(new_height * (1 - distance / max_distance))
        else:
            position = 0

        # set up vertices after warp
        pts2 = np.float32([[left_base_2, position], [right_base_2, position],
                           [left_base_2, new_height - 1], [right_base_2, new_height - 1]])

        # perspective transformation matrix
        M = cv2.getPerspectiveTransform(pts1, pts2)

        image_warped = cv2.warpPerspective(image, M, (half_new_width * 2, new_height), cv2.INTER_NEAREST, cv2.BORDER_REPLICATE)#, borderMode=cv2.BORDER_REPLICATE)
        mask_warped = cv2.warpPerspective(mask, M, (half_new_width * 2, new_height), cv2.INTER_NEAREST, cv2.BORDER_CONSTANT, 0)#, borderMode=cv2.BORDER_REPLICATE)
        #mask_warped[np.where(mask_warped < 255)] = 0

        # a secondary mask to mask those pixels beyond max_distance
        if distance < max_distance:
            mask_distance = np.zeros([new_height, 2*half_new_width], dtype='uint8')
            mask_distance[position:-1, :] = 255
            mask_warped = cv2.bitwise_and(mask_warped, mask_distance)

        self.image_warped = image_warped
        self.mask_warped = mask_warped
        self.warp_matrix = M

        return image_warped, mask_warped, M

    def line_filter(self, image, lower_lim, win_size=17, C=-20, rate=1.5, loop=3, gamma=2.0):
        # used median filter to delete potential lane markers and estimate a background with light condition
        median_img = cv2.medianBlur(image, 17)  # rough background with light condition (small dark area which < win_sire will not be detected)
        # small dark area can be shadows, correct the filtered image as below
        median_img[np.where(image < median_img)] = image[np.where(image < median_img)] # final background with light condition

        # the difference below estimates everything which stand out from the background regardless of its local light condition
        diff_img = image - median_img
        lane_marker_index = cv2.adaptiveThreshold(diff_img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                                  cv2.THRESH_BINARY, win_size, C)
        return diff_img, lane_marker_index

    # note! this one is no more used it basically does the same thing but much less efficient then that one above ^^
    def line_filter_old(self, image, lower_lim, win_size=17, C=-20, rate=1.5, loop=3, gamma=2.0):
        # process the image to compensate different light condition and finally convert it to binary image.
        # (the process add more light in darker area than the lighter area, than normalize histogram of the image to the
        # mean value as before image warp)
        # Note: line markers are smaller than the win_size, so they wont be compensated. but if light spot is
        # also smaller, they wont be compensated too. (small shadows will still be compensated correctly)

        maximum_ori = np.max(image)
        #minimum_ori = np.min(image)
        #mean_ori = np.mean(image)

        image_new2 = image.copy()
        for n in range(loop):
            corrected_meidan_img = image_new2.copy()

            # median filter filter out all small darker and ligher pixels
            median_img = cv2.medianBlur(image_new2, win_size)

            # preserve all darker pixels
            corrected_meidan_img[np.where(image_new2 > median_img)] = median_img[np.where(image_new2 > median_img)]

            # calculate a opposite image, called add_light2
            add_light2 = np.ones_like(corrected_meidan_img) * np.max(corrected_meidan_img) - corrected_meidan_img
            add_light2 = cv2.normalize(add_light2.astype(float) ** gamma, None, np.min(add_light2) / rate,
                                       np.max(add_light2) / rate, cv2.NORM_MINMAX)
            # image_new2 = cv2.add(add_light2.astype('uint8'), image_new2)  # note: here need further investigation
            image_new2 = add_light2 + image_new2
            # minimum_ori = np.min(image_new2)
            image_new2 = cv2.normalize(image_new2, None, lower_lim, maximum_ori, cv2.NORM_MINMAX).astype('uint8')
            # note: or normalize to fix value like 0 to 255 and apply a global threshold?

        # convert to binary
        lane_marker_index2 = cv2.adaptiveThreshold(image_new2.astype('uint8'), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, win_size, C)
        # lane_marker_index = Median_c(image_new2, 255, 21, -30)
        #lane_marker_index2 = cv2.Canny(image_new2.astype('uint8'), 50, 100)

        return image_new2, lane_marker_index2

    def lines_to_original(self,lines):
        warp_matrix = self.warp_matrix

        lines_col1 = lines[:, 0:2]
        lines_col2 = lines[:, 2:4]

        M_inverse = np.linalg.inv(warp_matrix)
        lines_col1_append = lines_col1.reshape(1, -1, 2).astype(np.float32)
        lines_col2_append = lines_col2.reshape(1, -1, 2).astype(np.float32)
        lines_col1_original = cv2.perspectiveTransform(lines_col1_append, M_inverse)[0]
        lines_col2_original = cv2.perspectiveTransform(lines_col2_append, M_inverse)[0]

        return np.hstack([lines_col1_original, lines_col2_original])

    def lidar_interpolation(self, point1, depth_image):
        x_sample = self.x_sample
        y_sample = self.y_sample
        x_sample_deg = self.x_sample_deg
        y_sample_deg = self.y_sample_deg
        upper_lim = self.upper_lim
        focus = self.focus
        h_sample = self.h_sample
        v_sample = 300  # !!


        x0 = point1[0]
        y0 = point1[1]

        theta, phi = find_lidar_theta_phi_from_image(x0, y0, focus, h_sample//2, v_sample//2)

        x1, y1 = find_sample_from_angle(phi, theta, x_sample_deg, y_sample_deg, x_sample, upper_lim)
        # x_sample_deg, y_sample_deg, x_sample, upper_lim

        if x1 >= x_sample-1:
            x1 = x_sample-2
        if x1 < 0:
            x1 = 0
        if y1 >= y_sample-1:
            y1 = y_sample-2
        if y1 < 0:
            y1 = 0

        x2 = x1 + 1
        y2 = y1 + 1

        depth_y1_x1 = depth_image[y1, x1]
        depth_y1_x2 = depth_image[y1, x2]
        depth_y2_x1 = depth_image[y2, x1]
        depth_y2_x2 = depth_image[y2, x2]

        angle_0 = find_lidar_phi_from_coord(x1, x_sample, x_sample_deg)
        angle_1 = phi
        angle_2 = find_lidar_phi_from_coord(x2, x_sample, x_sample_deg)
        # print(theta/np.pi*180, phi/np.pi*180)
        # print(x1,y1)
        # print(angle_0, angle_1, angle_2)
        depth_y1_inter, sensitivity_y1_inter = interpolation(angle_0, angle_1, angle_2, depth_y1_x1, depth_y1_x2)
        depth_y2_inter, sensitivity_y2_inter = interpolation(angle_0, angle_1, angle_2, depth_y2_x1, depth_y2_x2)

        angle_0 = find_lidar_theta_from_coord(y1, y_sample_deg, upper_lim)  # +1 due to lidar deepergtav
        angle_1 = theta  # positive if is above the horizen
        angle_2 = find_lidar_theta_from_coord(y2, y_sample_deg, upper_lim)  # +1 due to lidar deepergtav

        depth_last_inter, sensitivity_last_inter = interpolation(angle_0, angle_1, angle_2, depth_y1_inter,
                                                                 depth_y2_inter)

        X, Y, Z = polar_to_cartesian(depth_last_inter, theta, phi)

        return X, Y, Z

    def line2D_to_world_coord(self, lines):
        warp_matrix = self.warp_matrix
        focus = self.focus
        depth_image = self.depth_image
        pitch = self.pitch
        roll = self.roll

        lines_col1 = lines[:, 0:2]
        lines_col2 = lines[:, 2:4]

        # reverse project to original image
        M_inverse = np.linalg.inv(warp_matrix)
        lines_col1_append = lines_col1.reshape(1, -1, 2).astype(np.float32)
        lines_col2_append = lines_col2.reshape(1, -1, 2).astype(np.float32)
        lines_col1_original = cv2.perspectiveTransform(lines_col1_append, M_inverse)[0]
        lines_col2_original = cv2.perspectiveTransform(lines_col2_append, M_inverse)[0]


        lines_col1_world = np.zeros((lines_col1.shape[0], lines_col1.shape[1] + 1))
        lines_col2_world = np.zeros((lines_col1.shape[0], lines_col1.shape[1] + 1))
        n=0
        for point1, point2 in zip(lines_col1_original, lines_col2_original):

            # calculate new coords in camera/lidar coord
            p1_x, p1_y, p1_z = self.lidar_interpolation(point1, depth_image)
            p2_x, p2_y, p2_z = self.lidar_interpolation(point2, depth_image)

            lines_col1_world[n] = [p1_x, p1_y, p1_z]
            lines_col2_world[n] = [p2_x, p2_y, p2_z]
            n +=1

        # eliminate vehicle roll pitch
        move_matrix = [0,0,0]
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(-pitch, -roll, 0)
        lines_col1_world_rotated = coords_ro_move(lines_col1_world, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='turn_first')
        lines_col2_world_rotated = coords_ro_move(lines_col2_world, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='turn_first')

        lines_world_rotated_2d = np.vstack([lines_col1_world_rotated[:,0], lines_col1_world_rotated[:,2],
                                            lines_col2_world_rotated[:,0], lines_col2_world_rotated[:,2]]).T

        return lines_world_rotated_2d


    def fit_line(self, lane_marker_index):
        # filter small noises
        lane_marker_index = cv2.medianBlur(lane_marker_index, 3)

        # detect lines
        lines_raw = cv2.HoughLinesP(lane_marker_index, rho=2, theta=np.pi / 180, threshold=40, minLineLength=40, maxLineGap=10)

        # initialize a black image to show results
        lane_marker_index = cv2.cvtColor(lane_marker_index, cv2.COLOR_GRAY2BGR)
        map_image = np.zeros((600,450,3), dtype='uint8')
        coord_trans = np.array([map_image.shape[1] // 2, map_image.shape[0], map_image.shape[1] // 2, map_image.shape[0]])
        show_factor = 15

        if lines_raw is not None:
            lines_raw = lines_raw[:, 0, :] # reduce a dimension from cv's output
            lines_all = self.line2D_to_world_coord(lines_raw)

            # show line_raw
            for line in lines_raw:
                cv2.line(lane_marker_index, (line[0], line[1]), (line[2], line[3]), (250, 0, 0), 2)


            #### Debug
            #for line in lines_all:
            #    line = (coord_trans - line * show_factor).astype(int)
            #    cv2.line(map_image, (line[0], line[1]), (line[2], line[3]),
            #             (250, 100, 100), 2)

            ###########################################################

            if self.lines_last is not None:
                tracked_index, lines_last_esti = line_tracker2(lines_all, self.lines_last, self.delta_yaw, self.move_x, self.move_z)
                lines = lines_all[tracked_index]

                #### Debug
                #for line in self.lines_last:
                #    line = (coord_trans - line * show_factor).astype(int)
                #    cv2.line(map_image, (line[0], line[1]), (line[2], line[3]),
                #             (100, 100, 250), 2)

                #for line in lines_last_esti:
                #    line = (coord_trans - line * show_factor).astype(int)
                #    cv2.line(map_image, (line[0], line[1]), (line[2], line[3]),
                #             (100, 250, 100), 2)

            else:
                return lane_marker_index, map_image, None, lines_all


            ###########################################################
            # cluster and fit cluster
            if lines.size !=0:
                t0 = time.time()
                cluster_index = connect_lines(lines) # cluster the lines from hough transformation
                print('cluster takes: ', time.time() - t0)

                fitted_lines, cluster_index_2, line_length = cluster_fit(lines, cluster_index) # form line markers
                lanes = esti_lane(fitted_lines, cluster_index_2)

            ###########################################################
                # draw lines
                if np.max(cluster_index) != 0:
                    color = cv2.normalize(cluster_index.astype('uint8'), None, 30, 180, cv2.NORM_MINMAX)
                    color = np.expand_dims(color, axis=1)
                    color2 = np.ones_like(color, dtype='uint8') * 255
                    color = np.concatenate((color,color2,color2), axis=2)
                    color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)
                else:
                    color = np.ones_like(cluster_index.astype('uint8')) * 180
                    color = np.expand_dims(color, axis=1)
                    color = np.expand_dims(color, axis=2)
                    color2 = np.ones_like(color, dtype='uint8') * 255
                    color = np.concatenate((color, color2, color2), axis=2)
                    color = cv2.cvtColor(color, cv2.COLOR_HSV2BGR)

                n = 0
                for line in lines:
                    line = (coord_trans - line * show_factor).astype(int)
                    cv2.line(map_image, (line[0], line[1]), (line[2], line[3]), (int(color[n, 0, 0]), int(color[n, 0, 1]), int(color[n, 0, 2])), 3)
                    n += 1

                for line2 in fitted_lines:
                    line2 = (coord_trans - line2 * show_factor).astype(int)
                    cv2.line(map_image, (line2[0], line2[1]), (line2[2], line2[3]), (100, 100, 100), 2)

                if lanes.size !=0:
                    for lane in lanes:
                        lane = (coord_trans - lane * show_factor).astype(int)
                        cv2.line(map_image, (lane[0], lane[1]), (lane[2], lane[3]), (100, 100, 100), 20)
                else:
                    print('NO LANE FOUND!!!')
                    lanes = None

            else:
                print('NO LINE LEFT AFTER TRACKING!!!')
                lanes = None

        else:
            print('NO LINE FITTED!!!')
            lanes = None
            lines_all = None

        return lane_marker_index, map_image, lanes, lines_all

    def run(self):

        if self.reference_x is None: # still waiting for data
            return None, None, None, None, None, None, None, None, None

        self.calculate_movement()
        self.roi()
        self.rotation_correction()
        image_warped, mask_warped, warp_matrix = self.warp()

        mean_r = self.mean_r
        mean_g = self.mean_g
        mean_b = self.mean_b

        # channel independently buffer and detection

        #image2 = image_warped.copy()
        #image2[..., 0], lane_marker_index_r = self.lane_filter(image_warped[:, :, 0], mean_r)
        #image2[..., 1], lane_marker_index_g = self.lane_filter(image_warped[:, :, 1], mean_g)
        #image2[..., 2], lane_marker_index_b = self.lane_filter(image_warped[:, :, 2], mean_b)

        #lane_marker_index_rg = cv2.bitwise_and(lane_marker_index_r, lane_marker_index_g)
        #lane_marker_index_w = cv2.bitwise_and(lane_marker_index_rg, lane_marker_index_b)
        #lane_marker_index_w = cv2.bitwise_and(lane_marker_index_w, mask_warped)

        #lane_marker_index_y = cv2.bitwise_and(lane_marker_index_rg, ~lane_marker_index_b)
        #lane_marker_index_y = cv2.bitwise_and(lane_marker_index_y, mask_warped)

        #image2[:, :, 0] = cv2.bitwise_and(image2[..., 0], mask_warped)
        #image2[:, :, 1] = cv2.bitwise_and(image2[..., 1], mask_warped)
        #image2[:, :, 2] = cv2.bitwise_and(image2[..., 2], mask_warped)

        # channel mixed buffer and detection
        # 1. yellow-blue image to detect yellow lines
        image_y = (image_warped[..., 0].astype(int) + image_warped[..., 1].astype(int)) / 2 - image_warped[..., 2].astype(int)
        image_y = (image_y + 255)/2
        image_y = image_y.astype('uint8')
        #image_y = cv2.normalize(image_y, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
        mean_y = (((mean_r + mean_g) / 2 - mean_b) + 255) / 2

        t0= time.time()
        image_y_balanced, lane_marker_index_y = self.line_filter(image_y, mean_y, C=-5)
        print('each lane filter takes: ', time.time() - t0)

        lane_marker_index_y = cv2.bitwise_and(lane_marker_index_y, mask_warped)
        image_y_balanced = cv2.bitwise_and(image_y_balanced, mask_warped)

        # 2. gray image to detect white lines
        image_gray = np.min(image_warped, axis=2)
        mean_gray = min(mean_r, mean_g, mean_b)
        image_gray_balanced, lane_marker_index_w = self.line_filter(image_gray, mean_gray, C=-7)

        lane_marker_index_w = cv2.bitwise_and(lane_marker_index_w, mask_warped)
        image_gray_balanced = cv2.bitwise_and(image_gray_balanced, mask_warped)

        lane_marker_index_w, map_image, lanes, lines_all = self.fit_line(cv2.bitwise_or(lane_marker_index_w, lane_marker_index_y))

        #if lanes is not None and self.lanes_last is not None:
        #    tracked_index, lane_marker_index_w = line_tracker(lanes, self.lanes_last, self.delta_yaw, self.move_x, self.move_z, lane_marker_index_w, flag_show=True)

        image_warped[..., 0] = cv2.bitwise_and(image_warped[...,0], mask_warped)
        image_warped[..., 1] = cv2.bitwise_and(image_warped[..., 1], mask_warped)
        image_warped[..., 2] = cv2.bitwise_and(image_warped[..., 2], mask_warped)

        ### additional test part
        #if lanes is not None:
        #    fill = np.zeros((lanes.shape[0]))-1.6
        #    lanes_3d_col1 = np.vstack([lanes[:, 0], fill, lanes[:, 1]]).T
        #    lanes_3d_col2 = np.vstack([lanes[:, 2], fill, lanes[:, 3]]).T
        #    lanes_3d_col1 = coords_roll_move(lanes_3d_col1, 0, 0, 0, self.pitch, self.roll, 0, mode='turn_first')
        #    lanes_3d_col2 = coords_roll_move(lanes_3d_col2, 0, 0, 0, self.pitch, self.roll, 0, mode='turn_first')
#
        #    lanes_3d = np.hstack([lanes_3d_col1, lanes_3d_col2])
        #    print(lanes)
        #    print(lanes_3d)
        #else:
        #    lanes_3d = None

        self.lines_last = lines_all

        return lanes, map_image, self.move_x, self.move_z, lane_marker_index_w, lane_marker_index_y, image_y_balanced, image_gray_balanced, image_warped

