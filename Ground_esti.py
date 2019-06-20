import numpy as np
import time
import math
import cv2
import statistics
from Lidar_tools import polar_to_cartesian, find_lidar_theta_phi_from_coord_Ma, find_lidar_phi_from_coord, find_lidar_theta_from_coord
from Lidar_tools_AG import coords_ro_move, build_roatation_matrix_3D


def ground_marker_2_image(ground_marker, reference_x, reference_y):
    marker_ground_ref = np.concatenate((np.expand_dims(reference_x[ground_marker], axis=1),
                                        np.expand_dims(reference_y[ground_marker], axis=1)), axis=1)
    return marker_ground_ref


class Ground_esti:
    def __init__(self, Lidar_info):
        self.depth_image = None
        self.ground_marker = None
        self.x_sample_deg = Lidar_info['x_sample_deg']
        self.y_sample_deg = Lidar_info['y_sample_deg']
        self.upper_lim = Lidar_info['upper_lim']
        self.y_sample = Lidar_info['y_sample']
        self.x_sample = Lidar_info['x_sample']
        self.pitch = None
        self.roll = None

    def input(self, depth_image):
        self.depth_image = depth_image


    def calculate_from_xy(self, image_x1, image_x2, image_y1, image_y2):
        depth_image_now = self.depth_image
        #y_sample = self.y_sample
        x_sample = self.x_sample
        y_sample_deg = self.y_sample_deg
        x_sample_deg = self.x_sample_deg
        upper_lim = self.upper_lim

        theta1 = find_lidar_theta_from_coord(image_y1, y_sample_deg, upper_lim)
        theta2 = find_lidar_theta_from_coord(image_y2, y_sample_deg, upper_lim)

        phi1 = find_lidar_phi_from_coord(image_x1, x_sample, x_sample_deg)
        phi2 = find_lidar_phi_from_coord(image_x2, x_sample, x_sample_deg)

        x1, y1, z1 = polar_to_cartesian(depth_image_now[image_y2, image_x1], theta2, phi1)
        x2, y2, z2 = polar_to_cartesian(depth_image_now[image_y2, image_x2], theta2, phi2)
        x3, y3, z3 = polar_to_cartesian(depth_image_now[image_y1, image_x1], theta1, phi1)

        vector1_x = (y2 - y1) * (z3 - z1) - (y3 - y1) * (z2 - z1)
        vector1_y = (z2 - z1) * (x3 - x1) - (z3 - z1) * (x2 - x1)
        vector1_z = (x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1)
#        assert (vector1_y > 0)
        roll = -math.atan(vector1_x / vector1_y)
        pitch = -math.atan(vector1_z / vector1_y)
        return roll, pitch


    def roll_pitch(self): # moved to tools

        y_sample = self.y_sample
        x_sample = self.x_sample

        # take 3x3 samples from left right center, forms 3 vector
        y1 = y_sample - 2  # up
        y2 = y_sample - 1  # down

        x1 = x_sample // 2 - 2  # left
        x2 = x_sample // 2 + 1  # right

        x3 = x_sample // 2 - 21  # left
        x4 = x_sample // 2 - 20  # right

        x5 = x_sample // 2 + 20  # left
        x6 = x_sample // 2 + 21  # right

        x_list = [[x1,x2],[x3,x4],[x5,x6]]
        roll_list = []
        pitch_list = []
        for n in range(3):
            x_a = x_list[n][0]
            x_b = x_list[n][1]
            roll, pitch = self.calculate_from_xy(x_a, x_b, y1, y2)
            roll_list.append(roll)
            pitch_list.append(pitch)

        self.roll = statistics.median(roll_list)
        self.pitch = statistics.median(pitch_list)



    #def roll_pitch2(self): # moved to tools
#
    #    y_sample = self.y_sample
    #    x_sample = self.x_sample
    #    y_sample_deg = self.y_sample_deg
    #    x_sample_deg = self.x_sample_deg
    #    upper_lim = self.upper_lim
    #    depth_image_now = self.depth_image
#
    #    # points' xy to measure roll and pitch
    #    point_x1 = x_sample // 2 - 2 # left
    #    point_x2 = x_sample // 2 + 1 # right
    #    point_y1 = y_sample - 3  # up
    #    point_y2 = y_sample - 1  # down
#
    #    # same as
    #    # if x + 1 <= x_sample / 2:  # left side
    #    #     phi_last = (x_sample_deg * (x_sample / 2 - x) - x_sample_deg / 2) / 180 * np.pi
    #    # else:  # right side
    #    #     phi_last = - (x_sample_deg * (x + 1 - x_sample / 2) - x_sample_deg / 2) / 180 * np.pi
    #    # theta_last = (upper_lim + y * y_sample_deg) / 180 * np.pi
#
    #    point_x1_phi = (x_sample_deg * (x_sample / 2 - point_x1) - x_sample_deg / 2) / 180 * np.pi
    #    point_x2_phi = - (x_sample_deg * (point_x2 + 1 - x_sample / 2) - x_sample_deg / 2) / 180 * np.pi
    #    point_y1_theta = (upper_lim + point_y1 * y_sample_deg) / 180 * np.pi
    #    point_y2_theta = (upper_lim + point_y2 * y_sample_deg) / 180 * np.pi
#
    #    point_1_depth = depth_image_now[point_y1, point_x1]
    #    point_2_depth = depth_image_now[point_y1, point_x2]
    #    point_3_depth = depth_image_now[point_y2, point_x1]
    #    point_4_depth = depth_image_now[point_y2, point_x2]
#
    #    # left side, parameters for pitch and roll, all in camera coord, not vehicle coord
    #    # for equation pitch
    #    delta_y_pitch = point_3_depth * math.cos(point_y2_theta) - point_1_depth * math.cos(point_y1_theta)
    #    delta_x_pitch = point_3_depth * math.sin(point_y2_theta) * math.sin(
    #        point_x1_phi) - point_1_depth * math.sin(point_y1_theta) * math.sin(point_x1_phi)
    #    delta_z_pitch = point_3_depth * math.sin(point_y2_theta) * math.cos(
    #        point_x1_phi) - point_1_depth * math.sin(point_y1_theta) * math.cos(point_x1_phi)
    #    # for equation roll
    #    delta_y_roll = point_2_depth * math.cos(point_y1_theta) - point_1_depth * math.cos(point_y1_theta)
    #    delta_z_roll = point_2_depth * math.sin(point_y1_theta) * math.cos(point_x2_phi) - point_1_depth * math.sin(
    #        point_y1_theta) * math.cos(point_x1_phi)
    #    delta_x_roll = point_2_depth * math.sin(point_y1_theta) * math.sin(point_x2_phi) - point_1_depth * math.sin(
    #        point_y1_theta) * math.sin(point_x1_phi)
#
    #    pitch_13 = math.atan((delta_y_pitch * delta_x_roll - delta_y_roll * delta_x_pitch) / (
    #                delta_z_pitch * delta_x_roll - delta_x_pitch * delta_z_roll))
    #    roll_12 = math.atan((delta_y_pitch * delta_z_roll - delta_y_roll * delta_z_pitch) / (
    #                delta_z_pitch * delta_x_roll - delta_x_pitch * delta_z_roll))
#
    #    ############################################ the other two combinations in these 4 points
    #    delta_y_pitch = point_4_depth * math.cos(point_y2_theta) - point_2_depth * math.cos(point_y1_theta)
    #    delta_x_pitch = point_4_depth * math.sin(point_y2_theta) * math.sin(
    #        point_x2_phi) - point_2_depth * math.sin(point_y1_theta) * math.sin(point_x2_phi)
    #    delta_z_pitch = point_4_depth * math.sin(point_y2_theta) * math.cos(
    #        point_x2_phi) - point_2_depth * math.sin(point_y1_theta) * math.cos(point_x2_phi)
#
    #    delta_y_roll = point_3_depth * math.cos(point_y2_theta) - point_4_depth * math.cos(point_y2_theta)
    #    delta_z_roll = point_3_depth * math.sin(point_y2_theta) * math.cos(point_x1_phi) - point_4_depth * math.sin(
    #        point_y2_theta) * math.cos(point_x2_phi)
    #    delta_x_roll = point_3_depth * math.sin(point_y2_theta) * math.sin(point_x1_phi) - point_4_depth * math.sin(
    #        point_y2_theta) * math.sin(point_x2_phi)
#
    #    pitch_24 = math.atan((delta_y_pitch * delta_x_roll - delta_y_roll * delta_x_pitch) / (
    #                delta_z_pitch * delta_x_roll - delta_x_pitch * delta_z_roll))
    #    roll_34 = math.atan((delta_y_pitch * delta_z_roll - delta_y_roll * delta_z_pitch) / (
    #                delta_z_pitch * delta_x_roll - delta_x_pitch * delta_z_roll))
#
    #    self.pitch = (pitch_24 + pitch_13) / 2
    #    self.roll = (roll_12 + roll_34) / 2


    def check_ground(self):
        depth_image = self.depth_image
        x_sample_deg = self.x_sample_deg
        y_sample_deg = self.y_sample_deg
        upper_lim = self.upper_lim
        pitch = self.pitch
        roll = self.roll
        y_sample, x_sample = self.y_sample, self.x_sample

        # thresholds are angles between lidar samples, that should not be exceeded
        threshold_x = 7 / 180 * np.pi
        threshold_y = 10 / 180 * np.pi

        # a mask to show where is ground
        ground_marker = np.zeros_like(depth_image, dtype='bool')
        # assume and set the two front nearest samples are ground
        ground_marker[-1, x_sample // 2 - 1: x_sample // 2 + 1] = True


        row_lidar, col_lidar = (np.indices((y_sample, x_sample))).astype('float32')
        theta, phi = find_lidar_theta_phi_from_coord_Ma(row_lidar, col_lidar, x_sample, x_sample_deg, y_sample_deg, upper_lim)
        depth_x, depth_y, depth_z = polar_to_cartesian(depth_image, theta, phi)

        depth_x = np.expand_dims(depth_x, axis=2)
        depth_y = np.expand_dims(depth_y, axis=2)
        depth_z = np.expand_dims(depth_z, axis=2)
        all_points = np.concatenate((depth_x, depth_y, depth_z), axis=2).reshape(-1, 3)

        # compensation vehicle roll pitch
        move_matrix = np.array([0,0,0])
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(-pitch, -roll, 0)
        all_points = coords_ro_move(all_points, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                           mode='turn_first')

        depth_x = all_points[:, 0].reshape(y_sample, x_sample)
        depth_y = all_points[:, 1].reshape(y_sample, x_sample)
        depth_z = all_points[:, 2].reshape(y_sample, x_sample)

        # check the lowest lidar sample in x direction, seperatly check left and right side
        left_delta_x = depth_x[-1, 0:x_sample // 2 - 1] - depth_x[-1, 1:x_sample // 2]
        left_delta_y = depth_y[-1, 0:x_sample // 2 - 1] - depth_y[-1, 1:x_sample // 2]
        left_delta_z = depth_z[-1, 0:x_sample // 2 - 1] - depth_z[-1, 1:x_sample // 2]
        right_delta_x = depth_x[-1, x_sample // 2 + 1:] - depth_x[-1, x_sample // 2:- 1]
        right_delta_y = depth_y[-1, x_sample // 2 + 1:] - depth_y[-1, x_sample // 2:- 1]
        right_delta_z = depth_z[-1, x_sample // 2 + 1:] - depth_z[-1, x_sample // 2:- 1]

        left_slope = np.arctan(left_delta_y / np.sqrt(left_delta_x ** 2 + left_delta_z ** 2))
        right_slope = np.arctan(right_delta_y / np.sqrt(right_delta_x ** 2 + right_delta_z ** 2))

        # check from center to left end
        x_left = left_slope.size
        for x in range(left_slope.size-1,-1,-1):
            if left_slope[x] < threshold_x:
                ground_marker[-1, x] = 1
                #record the left end
                if x < x_left:
                    x_left = x
            else:
                break

        # check from center to right end
        offset = x_sample // 2 + 1
        x_right = -1
        for x in range(right_slope.size):
            if right_slope[x] < threshold_x:
                ground_marker[-1, offset + x] = 1
                if x > x_right:
                    x_right = x
            else:
                break
        x_right = offset + x_right

        # check in y direction
        delta_x = depth_x[0:y_sample - 1, :] - depth_x[1:y_sample, :]
        delta_y = depth_y[0:y_sample - 1, :] - depth_y[1:y_sample, :]
        delta_z = depth_z[0:y_sample - 1, :] - depth_z[1:y_sample, :]
        y_slope = np.arctan(delta_y / np.sqrt(delta_x ** 2 + delta_z ** 2))

        for x in range(x_left, x_right+1, 1):
            for y in range(y_sample-2,-1,-1):
                if y_slope[y, x] < threshold_y:
                    ground_marker[y, x] = 1
                else:
                    break

        self.ground_marker = ground_marker


    def run(self):
        self.roll_pitch()
        self.check_ground()
        return self.ground_marker, self.pitch, self.roll