import cv2
import time
import math
import statistics
import autograd.numpy as np
#import numpy as np
from autograd import grad
import matplotlib.pyplot as plt
import itertools
import random

from Lane_finder import line_compare
from Lidar_tools_AG import coords_ro_move_2D, build_rotation_matrix_2D

def to_lane_dist(lanes_col1, lanes_col2):
    # use middle point to calcu dist_y?
    sample_num = lanes_col1.shape[0]
    dist_y = np.min(abs(np.hstack((lanes_col1[:, 1].reshape(sample_num, 1), lanes_col2[:, 1].reshape(sample_num, 1)))), axis=1)  # nearest y in each lane

    theta = np.arctan((lanes_col1[:, 0] - lanes_col2[:, 0]) / (lanes_col2[:, 1] - lanes_col1[:, 1]))
    rho = abs(lanes_col1[:, 0] * np.cos(theta) + lanes_col1[:, 1] * np.sin(theta))  # offset to lane
    dist = dist_y + rho
    return dist


class lane_controler:
    def __init__(self):
        #self.warp_matrix = None
        self.lanes_now = None
        self.lanes_last = None
        self.order = None
        self.selected_lane_last = None
        self.selected_lane_now = None
        self.initialized = 0
        self.keep_last_lane_failed_count = 0

        self.delta_yaw = 0
        self.speed_last = 0
        self.speed_now = 0

        self.move_x = 0
        self.move_z = 0

        self.image = None

        #self.change_coord = np.array([225, 600])  # refer to Lane_finder
        self.change_coord2 = np.array([225, 600, 225, 600])

        # for EKF
        self.control_noise = np.array([0.1, 0.05, 0.05])
        self.measure_noise = np.array([[0.05, 0],
                                       [0, 0.05]])
        self.P = None
        self.x = None
        self.I = np.array([[1, 0], [0, 1]])
        self.EKF_lane_last = None

    def input(self, lanes, image, delta_yaw, move_x, move_z, v, acc, yaw,dt):
        #self.warp_matrix = warp_matrix
        self.lanes_now = lanes
        self.image = image
        self.delta_yaw = delta_yaw
        self.move_z = move_z
        self.move_x = move_x
        self.v = v
        self.acc = acc
        self.yaw = yaw
        self.dt_in_game = dt

    def build_line_from_theta_rho(self, theta, rho):
        x0 = math.cos(theta) * rho
        y0 = math.sin(theta) * rho
        length = 10
        x1 = x0 - length * math.sin(theta)
        y1 = y0 + length * math.cos(theta)
        return np.array([x0, y0, x1, y1])

    def control_inputs_to_rho(self, u):
        v, acc, yaw = u[0], u[1], u[2]
        x1, z1, x2, z2 = self.EKF_lane_last
        dt_in_game = self.dt_in_game

        if yaw != 0:
            #  simple ver. only using the yaw to estimate, in world coord: x,y,z refer to left, up, front
            horizontal_turn_radius = (v * dt_in_game + acc * dt_in_game ** 2 / 2) / yaw
            move_z = horizontal_turn_radius * np.sin(yaw)
            move_x = horizontal_turn_radius * (1 - np.cos(yaw))
        else:
            move_z = (v * dt_in_game + acc * dt_in_game ** 2 / 2)
            move_x = 0

        rotation_martix = np.array([[np.cos(yaw), -np.sin(yaw)],
                                    [np.sin(yaw), np.cos(yaw)]])
        move_matrix = np.array([move_x, move_z])
        points_col = np.array([[x1, z1], [x2, z2]])
        points_col_new = np.dot(points_col - move_matrix, np.linalg.inv(rotation_martix))

        theta = np.arctan((points_col_new[0, 0] - points_col_new[1, 0]) / (points_col_new[1, 1] - points_col_new[0, 1]))
        rho = points_col_new[0, 0] * np.cos(theta) + points_col_new[0, 1] * np.sin(theta)  # offset to lane
        return rho

    def EKF_init(self, lanes_now):
        theta = np.arctan((lanes_now[0] - lanes_now[2]) / (lanes_now[3] - lanes_now[1]))
        rho = lanes_now[0] * np.cos(theta) + lanes_now[1] * np.sin(theta)

        self.x = np.array([theta, rho])
        self.P = self.measure_noise


    def EKF_predict(self):
        move_x = self.move_x
        move_z = self.move_z
        delta_yaw = self.delta_yaw
        EKF_lane_last = self.EKF_lane_last

        EKF_lane_last_col1 = EKF_lane_last[0:2]
        EKF_lane_last_col2 = EKF_lane_last[2:4]

        # predict in cartesian coord
        move_matrix = np.array([move_x, move_z])
        rotation_martix = build_rotation_matrix_2D(delta_yaw)
        EKF_lane_now_esti_col1 = coords_ro_move_2D(EKF_lane_last_col1, move_matrix, rotation_martix,
                                                        mode='move first')
        EKF_lane_now_esti_col2 = coords_ro_move_2D(EKF_lane_last_col2, move_matrix, rotation_martix,
                                                        mode='move first')

        EKF_lane_now_esti = np.hstack((EKF_lane_now_esti_col1, EKF_lane_now_esti_col2))

        # change to rho theta space
        theta = np.arctan((EKF_lane_now_esti[0] - EKF_lane_now_esti[2]) / (
                    EKF_lane_now_esti[3] - EKF_lane_now_esti[1]))
        rho = EKF_lane_now_esti[0] * np.cos(theta) + EKF_lane_now_esti[1] * np.sin(theta)  # offset to lane

        self.x = np.array([theta, rho])
        Qj = grad(self.control_inputs_to_rho)

        # predict noise
        u = np.array([self.v, self.acc, self.yaw])
        rho_noise = np.sum(np.abs(Qj(u) * self.control_noise))
        theta_nois = self.control_noise[2]
        self.P += np.array([[theta_nois, 0],
                            [0, rho_noise]])


    def EKF_update(self, tracked_lane):
        theta = np.arctan((tracked_lane[0] - tracked_lane[2]) / (tracked_lane[3] - tracked_lane[1]))
        rho = tracked_lane[0] * np.cos(theta) + tracked_lane[1] * np.sin(theta)  # offset to lane

        z = np.array([theta, rho])
        P = self.P

        # update
        y = z - self.x
        S = P + self.measure_noise
        K = np.dot(P, np.linalg.inv(S))
        self.x += np.dot(K, y)
        self.P = np.dot((self.I - K), P)


    def lane_tracker(self, order):

        if order == 'keep last lane':
            lanes_now = self.lanes_now
            move_x = self.move_x
            move_z = self.move_z
            delta_yaw = self.delta_yaw
            selected_lane_last = self.selected_lane_last

            selected_lane_last_col1 = selected_lane_last[0:2]
            selected_lane_last_col2 = selected_lane_last[2:4]

            # move and rotate a selected_lane_last to current time
            move_matrix = np.array([move_x, move_z])
            rotation_martix = build_rotation_matrix_2D(delta_yaw)
            selected_lane_now_esti_col1 = coords_ro_move_2D(selected_lane_last_col1, move_matrix, rotation_martix,
                                                            mode='move first')
            selected_lane_now_esti_col2 = coords_ro_move_2D(selected_lane_last_col2, move_matrix, rotation_martix,
                                                            mode='move first')

            selected_lane_now_esti = np.hstack((selected_lane_now_esti_col1, selected_lane_now_esti_col2))

            n = 0
            selected_index = []
            for lane_now in lanes_now:
                # note: not use EKF_line to compare, since position information of a lane is lost in EKF
                tracked = line_compare(selected_lane_now_esti, lane_now, theta_thresh=np.pi/180*10, rho_thresh=1)
                if tracked == 1:
                    selected_index.append(n)
                n +=1

            if len(selected_index) == 1:
                selected_lane = lanes_now[selected_index,:][0]

                self.EKF_predict()
                self.EKF_update(selected_lane)
                return selected_lane
            elif len(selected_index) == 0:
                return None
            else:
                # multiple candidates, select the nearest one
                candidates = lanes_now[selected_index, :]
                dist = to_lane_dist(candidates[:,0:2], candidates[:,2:4])
                selected_lane = candidates[np.argmin(dist), :]

                self.EKF_predict()
                self.EKF_update(selected_lane)
                return selected_lane


        elif order == 'change to right nearest lane':
            lanes_now = self.lanes_now
            lanes_now_col1 = lanes_now[:, 0:2]
            lanes_now_col2 = lanes_now[:, 2:4]

            sample_num = lanes_now_col1.shape[0]
            stack_y = np.hstack((lanes_now_col1[:, 1].reshape(sample_num, 1), lanes_now_col2[:, 1].reshape(sample_num, 1)))
            index_lower_y = np.argmin(abs(stack_y), axis=1)
            stack_x = np.hstack((lanes_now_col1[:, 0].reshape(sample_num,1), lanes_now_col2[:, 0].reshape(sample_num,1)))
            lower_x = stack_x[list(range(sample_num)),index_lower_y]

            lanes_right = lanes_now[lower_x>0,:]
            if lanes_right.size !=0:
                dist = to_lane_dist(lanes_right[:,0:2], lanes_right[:,2:4])
                selected_lane = lanes_now[np.argmin(dist), :]

                self.EKF_init(selected_lane)
                return selected_lane
            else:
                return None

        elif order == 'change to left nearest lane':
            lanes_now = self.lanes_now
            lanes_now_col1 = lanes_now[:, 0:2]
            lanes_now_col2 = lanes_now[:, 2:4]

            sample_num = lanes_now_col1.shape[0]
            stack_y = np.hstack((lanes_now_col1[:, 1].reshape(sample_num, 1), lanes_now_col2[:, 1].reshape(sample_num, 1)))
            index_lower_y = np.argmin(abs(stack_y), axis=1)
            stack_x = np.hstack((lanes_now_col1[:, 0].reshape(sample_num, 1), lanes_now_col2[:, 0].reshape(sample_num, 1)))
            lower_x = stack_x[list(range(sample_num)), index_lower_y]

            lanes_left = lanes_now[lower_x < 0, :]
            if lanes_left.size !=0:
                dist = to_lane_dist(lanes_left[:,0:2], lanes_left[:,2:4])
                selected_lane = lanes_now[np.argmin(dist), :]

                self.EKF_init(selected_lane)
                return selected_lane
            else:
                return None


        elif order == 'pick a nearest lane':
            lanes_now = self.lanes_now
            lanes_now_col1 = lanes_now[:, 0:2]
            lanes_now_col2 = lanes_now[:, 2:4]

            dist = to_lane_dist(lanes_now_col1, lanes_now_col2)
            selected_lane = lanes_now[np.argmin(dist), :]

            # initialize a new EKF filter
            self.EKF_init(selected_lane)
            return selected_lane

        elif order == 'use inertial lane':
            selected_lane_last = self.selected_lane_last
            move_x = self.move_x
            move_z = self.move_z
            delta_yaw = self.delta_yaw

            selected_lane_last_col1 = selected_lane_last[0:2]
            selected_lane_last_col2 = selected_lane_last[2:4]

            move_matrix = np.array([move_x, move_z])
            rotation_martix = build_rotation_matrix_2D(delta_yaw)

            selected_lanes_now_esti_col1 = coords_ro_move_2D(selected_lane_last_col1, move_matrix, rotation_martix,
                                                             mode='move first')
            selected_lanes_now_esti_col2 = coords_ro_move_2D(selected_lane_last_col2, move_matrix, rotation_martix,
                                                             mode='move first')
            selected_lane_now = np.hstack((selected_lanes_now_esti_col1, selected_lanes_now_esti_col2))

            self.EKF_predict()
            return selected_lane_now
        else:
            print('order not correct')
            assert(0)


    def generate_order(self, order_override=None):
        lanes_now = self.lanes_now

        if lanes_now is not None:
            if self.initialized == 1 or order_override == 'pick a nearest lane':
                order = 'pick a nearest lane'
                selected_lane_now =self.lane_tracker(order)
                if selected_lane_now is not None:
                    return selected_lane_now
                else:
                    assert(0)  # cant pick a lane somehow, self.initialized will not change to 2 in the next call
            else:
                order = 'keep last lane'
                selected_lane_now = self.lane_tracker(order)
                if selected_lane_now is not None:
                    self.keep_last_lane_failed_count = 0
                    return selected_lane_now
                else:
                    self.keep_last_lane_failed_count += 1

                    if self.keep_last_lane_failed_count < 10:
                        # keep to inertial lane
                        selected_lane_now = self.lane_tracker('use inertial lane')
                        if max(selected_lane_now[1], selected_lane_now[3]) < 5:  # at least 1m away
                            selected_lane_now = self.generate_order(order_override='pick a nearest lane')
                    else:
                        selected_lane_now = self.generate_order(order_override='pick a nearest lane')

                    if selected_lane_now is not None:
                        return selected_lane_now
                    else:
                        assert (0),('there are lanes, but none selected')

        else:
            if self.initialized == 2: # no lane input, but older data is available
                # keep to inertial lane
                selected_lane_now = self.lane_tracker('use inertial lane')
                return selected_lane_now
            else:
                return None


    def run(self):
        if self.initialized == 0:
            if self.lanes_now is not None: # the very first lane data received
                self.initialized = 1  # state: first data ever, no older data available
            else:  # still waiting for data
                return self.image, None

        elif self.initialized == 1: # the second time it runs
            if self.lanes_last is not None:
                self.initialized = 2  # state: at least for 2 times function runs, older data available


        selected_lane_now = self.generate_order()
        EKF_lane = None
        # shoe info on image
        if selected_lane_now is not None:
            # calculate info:
            theta = np.arctan((selected_lane_now[0] - selected_lane_now[2]) / (selected_lane_now[3] - selected_lane_now[1]))
            rho = selected_lane_now[0] * np.cos(theta) + selected_lane_now[1] * np.sin(theta)  # offset to lane

            # print:
            print_lane = self.change_coord2 - (selected_lane_now * 15).astype(int)
            new_img = cv2.line(self.image.copy(), (print_lane[0], print_lane[1]), (print_lane[2], print_lane[3]), (255, 0, 0), 10)

            font = cv2.FONT_HERSHEY_SIMPLEX
            text = 'angle: {} offset: {}'.format(round(theta,2), round(rho,2))
            cv2.putText(new_img, text, (10, 50), font, 1, (255, 150,150), 1, cv2.LINE_AA)

            # draw EKF_lane
            theta, rho = self.x
            EKF_lane = self.build_line_from_theta_rho(theta, rho)

            print_lane = self.change_coord2 - (EKF_lane * 15).astype(int)
            new_img = cv2.line(new_img, (print_lane[0], print_lane[1]), (print_lane[2], print_lane[3]),
                               (0, 255, 0), 10)
            text = 'angle: {} offset: {}'.format(round(theta, 2), round(rho, 2))
            cv2.putText(new_img, text, (10, 100), font, 1, (150, 255, 150), 1, cv2.LINE_AA)
        else:
            new_img = self.image

        ## update last values:
        self.selected_lane_last = selected_lane_now
        self.EKF_lane_last = EKF_lane
        if self.lanes_now is not None:
            # update using real new value
            self.lanes_last = self.lanes_now

        else:
            # update using inertial value
            move_x = self.move_x
            move_z = self.move_z
            delta_yaw = self.delta_yaw

            if self.lanes_last is not None:
                lanes_last = self.lanes_last
                lanes_last_col1 = lanes_last[:, 0:2]
                lanes_last_col2 = lanes_last[:, 2:4]
                move_matrix = np.array([move_x, move_z])
                rotation_matrix = build_rotation_matrix_2D(delta_yaw)
                lanes_now_esti_col1 = coords_ro_move_2D(lanes_last_col1, move_matrix, rotation_matrix, mode='move first')
                lanes_now_esti_col2 = coords_ro_move_2D(lanes_last_col2, move_matrix, rotation_matrix, mode='move first')
                self.lanes_last = np.hstack((lanes_now_esti_col1, lanes_now_esti_col2))

            #if self.selected_lane_last is not None:
            #    selected_lane_last = self.selected_lane_last
            #    selected_lane_last_col1 = selected_lane_last[0:2]
            #    selected_lane_last_col2 = selected_lane_last[2:4]
            #    selected_lanes_now_esti_col1 = move_roatation_2d(selected_lane_last_col1, move_x, move_z, delta_yaw, mode='move first')
            #    selected_lanes_now_esti_col2 = move_roatation_2d(selected_lane_last_col2, move_x, move_z, delta_yaw, mode='move first')
            #    self.selected_lane_last = np.hstack((selected_lanes_now_esti_col1, selected_lanes_now_esti_col2))

        return new_img, EKF_lane








