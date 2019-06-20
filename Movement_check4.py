import numpy as np
from scipy.spatial import distance
import cv2
import math

import image_preprocess as IMG
import time

from Lidar_tools import edge_detect, polar_to_cartesian, interpolation, find_lidar_theta_phi_from_coord_Ma
from Lidar_tools_AG import build_roatation_matrix_3D, coords_ro_move



class Move_esti:
    def __init__(self, Lidar_info, Camera_info):
        self.plot = 1
        self.initialized = 0
        self.v_sample = Camera_info['v_sample']
        self.h_sample = Camera_info['h_sample']
        self.focus = Camera_info['focus']
        self.y_sample = Lidar_info['y_sample']
        self.x_sample = Lidar_info['x_sample']
        self.x_sample_deg = Lidar_info['x_sample_deg']
        self.y_sample_deg = Lidar_info['y_sample_deg']
        self.upper_lim = Lidar_info['upper_lim']
        self.lidar_range = Lidar_info['lidar_range']


        self.depth_image = []
        self.dt_in_game = []
        self.magnitude_image = []
        self.direction_image = []
        self.mag_read = []
        self.direc_read = None

        self.reference_y = None
        self.reference_y = None

        self.m_candidate_coord_last = np.array([])
        self.m_candidate2_coord_last = np.array([])
        self.m_tracked_coord_last = np.array([])

        self.m_tracked_spd_last = np.array([])
        self.m_candidate_spd_last = np.array([])
        self.m_candidate2_spd_last = np.array([])

        self.s_candidate_coord_last = np.array([])
        self.s_candidate2_coord_last = np.array([])
        self.s_tracked_coord_last = np.array([])

        self.pitch = []
        self.delta_roll = []
        self.delta_pitch = []
        self.delta_yaw = []
        self.speed = []
        self.acc = []

        self.move_z = []
        self.move_x = []
        self.move_y = []
        self.depth_x = []
        self.depth_y = []
        self.depth_z = []
        self.reference_y = None
        self.reference_x = None

        self.ground_marker = []
        self.track_times = 2
        self.count = 0


    def input(self, ground_marker, magnitude_image, direction_image, depth_image, dt_in_game,
               pitch, delta_roll, delta_pitch, delta_yaw, speed_now, acc):

        if len(self.depth_image) == self.track_times + 2:
            self.depth_image.pop(0)
            self.dt_in_game.pop(0)
            self.magnitude_image.pop(0)
            self.direction_image.pop(0)

            self.pitch.pop(0)
            self.delta_roll.pop(0)
            self.delta_pitch.pop(0)
            self.delta_yaw.pop(0)
            self.speed.pop(0)
            self.acc.pop(0)

            self.ground_marker.pop(0)

            self.move_z.pop(0)
            self.move_x.pop(0)
            self.move_y.pop(0)


        self.depth_image.append(depth_image)
        self.dt_in_game.append(dt_in_game)
        self.magnitude_image.append(magnitude_image)
        self.direction_image.append(direction_image)


        self.pitch.append(pitch)
        self.delta_roll.append(delta_roll)
        self.delta_pitch.append(delta_pitch)
        self.delta_yaw.append(delta_yaw)
        self.speed.append(speed_now)
        self.acc.append(acc)

        self.ground_marker.append(ground_marker)

        self.calculate_movement()


    def calculate_movement(self):
        #  simple ver. only using the yaw to estimate, in camera coord: x,y,z refer to front, width, height
        dt_in_game = self.dt_in_game[-1]
        acc = self.acc[-1]
        speed_last = self.speed[-1] - acc * dt_in_game
        delta_yaw = self.delta_yaw[-1]
        pitch = self.pitch[-1]

        if delta_yaw != 0:
            #  simple ver. only using the yaw to estimate, in camera coord: x,y,z refer to left, up, front
            horizontal_turn_radius = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) / delta_yaw
            self.move_z.append(horizontal_turn_radius * math.sin(delta_yaw) * math.cos(pitch))
            self.move_x.append(horizontal_turn_radius * (1 - math.cos(delta_yaw)))
            self.move_y.append((speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) * math.sin(pitch))
        else:
            self.move_z.append((speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) * math.cos(pitch))
            self.move_x.append(0)
            self.move_y.append((speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) * math.sin(pitch))


    def run(self):
        if self.initialized == 0:
            if len(self.depth_image) !=0:
                self.count +=1
            if self.count == 3:
                self.initialized = 1
            return None, None, None, None, None, None, None, None, None


        y_sample = self.y_sample
        x_sample = self.x_sample
        y_sample_deg = self.y_sample_deg
        x_sample_deg = self.x_sample_deg
        upper_lim = self.upper_lim
        depth_image = self.depth_image
        lidar_range = self.lidar_range
        focus = self.focus
        h_sample = self.h_sample
        v_sample = self.v_sample
        dt_in_game = self.dt_in_game

        ground_marker = self.ground_marker

        # initialize index and lists
        good_points_index = []
        spd_xyz = []
        spd_mag = []
        threshold_m = []
        threshold_s = []
        threshold_m_t_spd = []
        rotation_matrix_p_r = []
        rotation_matrix_y = []

        # xyz coord of each sample from the current(newest) data
        row_lidar, col_lidar = (np.indices((y_sample, x_sample))).astype('float32')

        theta, phi = find_lidar_theta_phi_from_coord_Ma(row_lidar, col_lidar, x_sample, x_sample_deg, y_sample_deg,
                                                        upper_lim)

        depth_x, depth_y, depth_z = polar_to_cartesian(depth_image[len(self.depth_image) - 1], theta, phi)

        coord_now = np.concatenate((np.expand_dims(depth_x, axis=2),
                                    np.expand_dims(depth_y, axis=2),
                                    np.expand_dims(depth_z, axis=2)), axis=2)

        # calculate 3d movement of points back in n times before, record the speed, threshold and index for good points
        for n in range(len(self.depth_image) - 1, 0, -1):

            # 2 derivative of depth image
            edge_y, edge_x = edge_detect(depth_image[n-1])

            # cartesian to image coord
            reference_y = (v_sample - 1) / 2 - depth_y * focus / depth_z
            reference_x = (h_sample - 1) / 2 - depth_x * focus / depth_z

            reference_y_round = np.round(reference_y)
            reference_x_round = np.round(reference_x)
            bad_index_y = (reference_y_round < 0) | (reference_y_round >= v_sample)
            bad_index_x = (reference_x_round < 0) | (reference_x_round >= h_sample)

            reference_y_round[bad_index_y] = 0
            reference_x_round[bad_index_x] = 0

            # read optical flow to find where the last positions were
            magnitude = self.magnitude_image[n][reference_y_round.astype(int), reference_x_round.astype(int)]
            direction = self.direction_image[n][reference_y_round.astype(int), reference_x_round.astype(int)]

            extend_row = magnitude * np.sin(direction / 180 * np.pi)
            extend_col = magnitude * np.cos(direction / 180 * np.pi)

            row_image_last = reference_y - extend_row  # row and col not rounded yet
            col_image_last = reference_x - extend_col


            # find lidar theta phi from those positions in image
            pixel_v_refer2center = row_image_last - v_sample // 2 + 0.5  # -: above, +: under
            pixel_h_refer2center = h_sample // 2 - col_image_last - 0.5  # +: left, -: right

            theta_last = np.arctan(pixel_v_refer2center / np.sqrt(focus ** 2 + pixel_h_refer2center ** 2)) + np.pi / 2
            phi_last = np.arctan(pixel_h_refer2center / focus)


            # find possible nearest upper left lidar point from phi and theta, this is one of the vertices for interpolation
            row_lidar_up_raw = ((theta_last / np.pi * 180 - upper_lim) // y_sample_deg).astype(int)

            col_lidar_left_raw = np.zeros_like(col_lidar)
            col_lidar_left_raw = np.where(phi_last / np.pi * 180 >= x_sample_deg / 2,  # left
                                          x_sample // 2 - (phi_last / np.pi * 180 - x_sample_deg / 2) // x_sample_deg - 2,
                                          col_lidar_left_raw)

            col_lidar_left_raw = np.where(phi_last / np.pi * 180 <= -x_sample_deg / 2,  # right
                                          x_sample // 2 + (-phi_last / np.pi * 180 - x_sample_deg / 2) // x_sample_deg,
                                          col_lidar_left_raw)

            col_lidar_left_raw = np.where((phi_last / np.pi * 180 < x_sample_deg / 2) &
                                          (phi_last / np.pi * 180 > -x_sample_deg / 2),  # near center
                                          x_sample // 2 - 1,
                                          col_lidar_left_raw)
            col_lidar_left_raw = col_lidar_left_raw.astype(int)

            # change those points which are out of lidar FOV into correct range. Theirs result cant be trusted.
            # All points will be interpolated due to matrix calculation, but results from bad points will be excluded later
            row_lidar_up = np.where(row_lidar_up_raw < 0, 0, row_lidar_up_raw)
            row_lidar_up = np.where(row_lidar_up + 1 > y_sample - 1, y_sample - 2, row_lidar_up)
            col_lidar_left = np.where(col_lidar_left_raw < 0, 0, col_lidar_left_raw)
            col_lidar_left = np.where(col_lidar_left + 1 > x_sample - 1, x_sample - 2, col_lidar_left)

            # the other vertices for interpolation
            row_lidar_down = row_lidar_up + 1
            col_lidar_right = col_lidar_left + 1

            # read depth of each vertices
            depth_ul = depth_image[n-1][row_lidar_up, col_lidar_left]
            depth_ur = depth_image[n-1][row_lidar_up, col_lidar_right]
            depth_dl = depth_image[n-1][row_lidar_down, col_lidar_left]
            depth_dr = depth_image[n-1][row_lidar_down, col_lidar_right]

            # interpolation
            theta_last_up, phi_last_left = \
                find_lidar_theta_phi_from_coord_Ma(row_lidar_up, col_lidar_left, x_sample, x_sample_deg, y_sample_deg,
                                                   upper_lim)
            theta_last_down, phi_last_right = \
                find_lidar_theta_phi_from_coord_Ma(row_lidar_down, col_lidar_right, x_sample, x_sample_deg, y_sample_deg,
                                                   upper_lim)

            depth_inter_u, sensitivity_up = interpolation(phi_last_left, phi_last, phi_last_right, depth_ul, depth_ur)
            depth_inter_d, sensitivity_down = interpolation(phi_last_left, phi_last, phi_last_right, depth_dl, depth_dr)

            depth_inter, sensitivity_center = interpolation(theta_last_up, theta_last, theta_last_down, depth_inter_u,
                                                               depth_inter_d)

            # to cartesian
            depth_x_last, depth_y_last, depth_z_last = polar_to_cartesian(depth_inter, theta_last, phi_last)

            # make indices for good point!
            obj_check1 = edge_x[row_lidar_up, col_lidar_left] * edge_x[row_lidar_up, col_lidar_right]
            obj_check2 = edge_x[row_lidar_down, col_lidar_left] * edge_x[row_lidar_down, col_lidar_right]
            obj_check3 = edge_y[row_lidar_up, col_lidar_left] * edge_y[row_lidar_down, col_lidar_left]
            obj_check4 = edge_y[row_lidar_up, col_lidar_right] * edge_y[row_lidar_down, col_lidar_right]

            depth_check1 = depth_image[n-1][row_lidar_up, col_lidar_left]
            depth_check2 = depth_image[n-1][row_lidar_up, col_lidar_right]
            depth_check3 = depth_image[n-1][row_lidar_down, col_lidar_left]
            depth_check4 = depth_image[n-1][row_lidar_down, col_lidar_right]


            good_points_index.append(((depth_image[n] < lidar_range) & # current points inside lidar range
                                # current points is not ground
                                (ground_marker[n] == 0) &
                                # 4 raw vertices inside lidar FOV
                                (row_lidar_up_raw >= 0) &
                                (row_lidar_up_raw + 1 <= y_sample - 1) &
                                (col_lidar_left_raw >= 0) &
                                (col_lidar_left_raw + 1 <= x_sample - 1) &
                                # 4 raw vertices located on the same surface of a object
                                (obj_check1 > -1) & (obj_check2 > -1) & (obj_check3 > -1) & (obj_check4 > -1) &
                                # 4 raw vertices inside lidar FOV
                                (depth_check1 < lidar_range) & (depth_check2 < lidar_range) &
                                (depth_check3 < lidar_range) & (depth_check4 < lidar_range) &
                                (~bad_index_y) & (~bad_index_x)).reshape(-1))

            # take out those good points.
            depth_x_last_temp = np.expand_dims(depth_x_last.reshape(-1), axis=1)
            depth_y_last_temp = np.expand_dims(depth_y_last.reshape(-1), axis=1)
            depth_z_last_temp = np.expand_dims(depth_z_last.reshape(-1), axis=1)
            points_last = np.concatenate((depth_x_last_temp, depth_y_last_temp, depth_z_last_temp), axis=1)  # refer to camera coord last

            depth_x = np.expand_dims(depth_x.reshape(-1), axis=1)
            depth_y = np.expand_dims(depth_y.reshape(-1), axis=1)
            depth_z = np.expand_dims(depth_z.reshape(-1), axis=1)
            points_now = np.concatenate((depth_x, depth_y, depth_z), axis=1)  # refer to camera coord now
            # (camera coord move in world coord)
            # change the coord system time last to time now by using self movement
            # ie, points_last refer to camera coord now

            move_matrix = np.array([self.move_x[n], self.move_y[n], self.move_z[n]])
            rotation_matrix_p_r_temp, rotation_matrix_y_temp = build_roatation_matrix_3D(self.delta_pitch[n], self.delta_roll[n],
                                                                               self.delta_yaw[n])

            rotation_matrix_p_r.append(rotation_matrix_p_r_temp)
            rotation_matrix_y.append(rotation_matrix_y_temp)
            points_last_new_coord = coords_ro_move(points_last, move_matrix, rotation_matrix_p_r_temp, rotation_matrix_y_temp,
                                                   mode='move_first')

            # speed relative to ground in current camera coord
            spd_xyz_temp = ((points_now - points_last_new_coord) / dt_in_game[n]) #.reshape(y_sample, x_sample, 3)
            spd_xyz.append(spd_xyz_temp)
            spd_mag.append(np.sqrt(np.sum(spd_xyz_temp ** 2, axis=1)))

            # save info from first loop
            if n == len(self.depth_image) - 1:
                spd_xyz_now = spd_xyz_temp.copy()
                self.reference_y = reference_y
                self.reference_x = reference_x

                # for check_flow(), debug
                self.depth_x = depth_x
                self.depth_y = depth_y
                self.depth_z = depth_z
                self.mag_read = magnitude
                self.direc_read = direction


            # build united sensitivity(of interpolation) matrix
            sensitivity_up = np.expand_dims(sensitivity_up, axis=2)
            sensitivity_down = np.expand_dims(sensitivity_down, axis=2)
            sensitivity_center = np.expand_dims(sensitivity_center, axis=2)
            sensitivity = np.concatenate((sensitivity_up, sensitivity_down, sensitivity_center), axis=2)

            # adaptive threshold,
            # m for detection moving points,
            # s for detection stationary points,
            # t for compare spd between tracking points
            threshold = (depth_image[n-1] * 0.05 + np.max(sensitivity, axis=2) * 0.95).reshape(-1)
            threshold_m.append(threshold)  # spd threshold
            threshold_s.append(threshold / 1) # spd threshold
            threshold_t = (1 + depth_image[n-1] * 0.05).reshape(-1)
            threshold_m_t_spd.append(threshold_t)  # + np.max(sensitivity, axis=2) * 0.95  # spd threshold

            # update values for next loop
            depth_x = depth_x_last
            depth_y = depth_y_last
            depth_z = depth_z_last


        # threshold current points
        stationary_index_now = spd_mag[0] < threshold_s[0]
        moving_index_now = spd_mag[0] > threshold_m[0]
        good_point_index_now = good_points_index[0]

        # threshold points and tracking in past n times
        stationary_tracked_level_index = stationary_index_now.copy()
        moving_tracked_level_index = moving_index_now.copy()
        good_point_level_index = good_point_index_now.copy()
        all_stationary_tracked_index = []
        all_moving_tracked_index = []
        all_good_point_tracked_index = []
        move_matrix = np.array([0,0,0])

        for n in range(1, len(self.depth_image) - 1, 1):
            stationary_index_temp = spd_mag[n] < threshold_s[n]
            moving_index_temp = spd_mag[n] > threshold_m[n]

            # change coord of the last spd n, so it can compare with n-1
            spd_xyz_temp = coords_ro_move(spd_xyz[n], move_matrix, rotation_matrix_p_r[n-1], rotation_matrix_y[n-1], mode='move_first')
            spd_diff = np.sqrt(np.sum((spd_xyz[n-1] - spd_xyz_temp) ** 2, axis=1))
            #spd_diff = np.zeros_like(spd_diff)
            same_spd_index_temp = (spd_diff < (threshold_m_t_spd[n-1] + threshold_m_t_spd[n]) / 2)

            stationary_tracked_level_index = stationary_tracked_level_index & stationary_index_temp
            moving_tracked_level_index = moving_tracked_level_index & moving_index_temp & same_spd_index_temp
            good_point_level_index = good_point_level_index & good_points_index[n]

            all_stationary_tracked_index.append(stationary_tracked_level_index)
            all_moving_tracked_index.append(moving_tracked_level_index)
            all_good_point_tracked_index.append(good_point_level_index)


        # assign index to each level of tracking
        s_tracked_index = all_stationary_tracked_index[-1]
        m_tracked_index = all_moving_tracked_index[-1]

        s_candidate2_index = (all_stationary_tracked_index[-2] ^ s_tracked_index)
        m_candidate2_index = (all_moving_tracked_index[-2] ^ m_tracked_index)

        s_candidate_index = (stationary_index_now ^ (s_candidate2_index | s_tracked_index))
        m_candidate_index = (moving_index_now ^ (m_candidate2_index | m_tracked_index))

        # exclude unvalid points
        s_tracked_index = s_tracked_index & all_good_point_tracked_index[-1]
        m_tracked_index = m_tracked_index & all_good_point_tracked_index[-1]

        s_candidate2_index = s_candidate2_index & all_good_point_tracked_index[-2]
        m_candidate2_index = m_candidate2_index & all_good_point_tracked_index[-2]

        s_candidate_index = s_candidate_index & all_good_point_tracked_index[0]
        m_candidate_index = m_candidate_index & all_good_point_tracked_index[0]

        # takes samples from index
        coord_now = coord_now.reshape(-1,3)
        spd_xyz_now = spd_xyz_now.reshape(-1,3)
        s_tracked_coord = coord_now[s_tracked_index]
        m_tracked_coord = coord_now[m_tracked_index]
        m_tracked_spd = spd_xyz_now[m_tracked_index]

        s_candidate2_coord = coord_now[s_candidate2_index]
        m_candidate2_coord = coord_now[m_candidate2_index]
        m_candidate2_spd = spd_xyz_now[m_candidate2_index]

        s_candidate_coord = coord_now[s_candidate_index]
        m_candidate_coord = coord_now[m_candidate_index]
        m_candidate_spd = spd_xyz_now[m_candidate_index]

        # add markers to show image
        marker = None
        if self.plot != 0:
            row = np.expand_dims(self.reference_y, axis=2)
            col = np.expand_dims(self.reference_x, axis=2)
            reference = (np.concatenate((col, row), axis=2)).reshape(-1,2)

            marker_s_tracked = reference[s_tracked_index]
            marker_s_candidate2 = reference[s_candidate2_index]
            marker_s_candidate = reference[s_candidate_index]

            marker_m_tracked = reference[m_tracked_index]
            marker_m_candidate2 = reference[m_candidate2_index]
            marker_m_candidate = reference[m_candidate_index]

            marker = {'marker_tracked': np.array(marker_m_tracked),
                      'marker_candidate2': np.array(marker_m_candidate2),
                      'marker_candidate': np.array(marker_m_candidate),
                      'marker_s_tracked': np.array(marker_s_tracked),
                      'marker_s_candidate2': np.array(marker_s_candidate2),
                      'marker_s_candidate': np.array(marker_s_candidate)}

        esti_coord = None
        excess = None
        lack = None

        self.m_tracked_coord_last = m_tracked_coord
        self.m_tracked_spd_last = m_tracked_spd
        self.m_candidate_coord_last = m_candidate_coord
        self.m_candidate_spd_last = m_candidate_spd
        self.m_candidate2_coord_last = m_candidate2_coord
        self.m_candidate2_spd_last = m_candidate2_spd

        self.s_tracked_coord_last = s_tracked_coord
        self.s_candidate_coord_last = s_candidate_coord
        self.s_candidate2_coord_last = s_candidate2_coord

        self.depth_image_last = self.depth_image

        # return the current data actually
        return m_tracked_spd, self.m_tracked_coord_last, self.s_tracked_coord_last, excess, lack, \
               self.reference_y, self.reference_x, marker, esti_coord



    # debug function
    def check_flow(self):
        move_matrix = np.array([-self.move_x[-1], -self.move_y[-1], -self.move_z[-1]])
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(-self.delta_pitch[-1], -self.delta_roll[-1],
                                                                           -self.delta_yaw[-1])

        depth_x = np.expand_dims(self.depth_x, axis=2)
        depth_y = np.expand_dims(self.depth_y, axis=2)
        depth_z = np.expand_dims(self.depth_z, axis=2)
        all_point_now = np.concatenate((depth_x,depth_y,depth_z), axis=2).reshape(-1,3)
        all_points_last_esti = coords_ro_move(all_point_now, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                              mode='turn_first')

        depth_x_last = all_points_last_esti[:,0].reshape(self.y_sample, self.x_sample)
        depth_y_last = all_points_last_esti[:,1].reshape(self.y_sample, self.x_sample)
        depth_z_last = all_points_last_esti[:,2].reshape(self.y_sample, self.x_sample)
        # cartesian to image coord
        row_image_last0 = (self.v_sample - 1) / 2 - depth_y_last * self.focus / depth_z_last
        col_image_last0 = (self.h_sample - 1) / 2 - depth_x_last * self.focus / depth_z_last

        flow_y = self.reference_y - row_image_last0
        flow_x = self.reference_x - col_image_last0

        mag_esti = np.sqrt(flow_x ** 2 + flow_y ** 2)
        mag_esti[np.where(mag_esti == 0)] = 1
        direc_esti = np.where(flow_y >= 0,
                              np.arccos(flow_x / mag_esti) / np.pi * 180,
                              np.arccos(- flow_x / mag_esti) / np.pi * 180 + 180)


        gamma = 3
        hsv = np.zeros((self.y_sample, self.x_sample, 3))
        hsv[..., 1] = 255
        hsv[..., 0] = self.direc_read / 2
        hsv[..., 2] = cv2.normalize(self.mag_read, None, 0, 1, cv2.NORM_MINMAX)
        hsv[..., 2] = 255 * hsv[..., 2] ** (1 / gamma)
        hsv = np.uint8(hsv)
        image_flow_cut = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        hsv_0 = np.zeros((mag_esti.shape[0], mag_esti.shape[1], 3))
        hsv_0[..., 1] = 255
        hsv_0[..., 0] = direc_esti / 2
        hsv_0[..., 2] = cv2.normalize(mag_esti, None, 0, 1, cv2.NORM_MINMAX)
        hsv_0[..., 2] = 255 * hsv_0[..., 2] ** (1 / gamma)
        hsv_0 = np.uint8(hsv_0)
        image_flow_cut_0 = cv2.cvtColor(hsv_0, cv2.COLOR_HSV2RGB)

        return image_flow_cut, image_flow_cut_0, self.mag_read, mag_esti, self.direc_read, direc_esti




