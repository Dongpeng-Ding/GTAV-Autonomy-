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


        self.depth_image_now = None
        self.depth_image_last = None
        self.dt_in_game = None
        self.magnitude_image = None
        self.direction_image = None
        self.mag_read = None
        self.direc_read = None

        self.reference_x = None
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

        self.pitch = None
        self.delta_roll = None
        self.delta_pitch = None
        self.delta_yaw = None
        self.speed_now = None
        self.acc = None

        self.move_z = None
        self.move_x = None
        self.move_y = None
        self.depth_x_now = None
        self.depth_y_now = None
        self.depth_z_now = None
        self.row_image_now = None
        self.col_image_now = None

        self.ground_marker = None


    def input(self, ground_marker, magnitude_image, direction_image, depth_image, dt_in_game,
               pitch, delta_roll, delta_pitch, delta_yaw, speed_now, acc):

        self.depth_image_now = depth_image
        self.dt_in_game = dt_in_game
        self.magnitude_image = magnitude_image
        self.direction_image = direction_image


        self.pitch = pitch
        self.delta_roll = delta_roll
        self.delta_pitch = delta_pitch
        self.delta_yaw = delta_yaw
        self.speed_now = speed_now
        self.acc = acc

        self.ground_marker = ground_marker

        self.calculate_movement()


    def calculate_movement(self):
        #  simple ver. only using the yaw to estimate, in camera coord: x,y,z refer to front, width, height
        dt_in_game = self.dt_in_game
        acc = self.acc
        speed_last = self.speed_now - acc * dt_in_game

        if self.delta_yaw != 0:
            #  simple ver. only using the yaw to estimate, in camera coord: x,y,z refer to left, up, front
            horizontal_turn_radius = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) / self.delta_yaw
            self.move_z = horizontal_turn_radius * math.sin(self.delta_yaw) * math.cos(self.pitch)
            self.move_x = horizontal_turn_radius * (1 - math.cos(self.delta_yaw))
            self.move_y = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) * math.sin(self.pitch)
        else:
            self.move_z = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) * math.cos(self.pitch)
            self.move_x = 0
            self.move_y = (speed_last * dt_in_game + acc * dt_in_game ** 2 / 2) * math.sin(self.pitch)



    def run(self):
        if self.initialized == 0:
            if self.depth_image_now is not None:
                self.initialized = 1
                self.depth_image_last = self.depth_image_now
            return None, None, None, None, None, None, None, None, None



        y_sample = self.y_sample
        x_sample = self.x_sample
        y_sample_deg = self.y_sample_deg
        x_sample_deg = self.x_sample_deg
        upper_lim = self.upper_lim
        depth_image_now = self.depth_image_now
        depth_image_last = self.depth_image_last
        lidar_range = self.lidar_range
        focus = self.focus
        h_sample = self.h_sample
        v_sample = self.v_sample
        dt_in_game = self.dt_in_game

        ground_marker = self.ground_marker

        m_candidate_coord_last = self.m_candidate_coord_last
        m_candidate2_coord_last = self.m_candidate2_coord_last
        m_tracked_coord_last = self.m_tracked_coord_last

        m_tracked_spd_last = self.m_tracked_spd_last
        m_candidate_spd_last = self.m_candidate_spd_last
        m_candidate2_spd_last = self.m_candidate2_spd_last

        s_candidate_coord_last = self.s_candidate_coord_last
        s_candidate2_coord_last = self.s_candidate2_coord_last
        s_tracked_coord_last = self.s_tracked_coord_last

        ### main calculation starts here
        edge_y, edge_x = edge_detect(depth_image_last)

        row_lidar, col_lidar = (np.indices((y_sample, x_sample))).astype('float32')

        # find lidar theta phi
        theta, phi = find_lidar_theta_phi_from_coord_Ma(row_lidar, col_lidar, x_sample, x_sample_deg, y_sample_deg,
                                                        upper_lim)

        # polar to cartesian camera coord
        depth_x, depth_y, depth_z = polar_to_cartesian(depth_image_now, theta, phi)

        # cartesian to image coord
        reference_y = (v_sample - 1) / 2 - depth_y * focus / depth_z
        reference_x = (h_sample - 1) / 2 - depth_x * focus / depth_z

        # read optical flow to find where the last positions were
        magnitude = self.magnitude_image[np.round(reference_y).astype(int), np.round(reference_x).astype(int)]
        direction = self.direction_image[np.round(reference_y).astype(int), np.round(reference_x).astype(int)]

        # use more samples to read optical flow
        #magnitude = (self.magnitude_image[np.round(reference_y).astype(int), np.round(reference_x).astype(int)] +
        #            self.magnitude_image[np.round(reference_y).astype(int), np.round(reference_x).astype(int)+1] +
        #            self.magnitude_image[np.round(reference_y).astype(int)+1, np.round(reference_x).astype(int)] +
        #            self.magnitude_image[np.round(reference_y).astype(int)+1, np.round(reference_x).astype(int)+1]) / 4
#
        #direction = (self.direction_image[np.round(reference_y).astype(int), np.round(reference_x).astype(int)] +
        #            self.direction_image[np.round(reference_y).astype(int), np.round(reference_x).astype(int)+1] +
        #            self.direction_image[np.round(reference_y).astype(int)+1, np.round(reference_x).astype(int)] +
        #            self.direction_image[np.round(reference_y).astype(int)+1, np.round(reference_x).astype(int)+1]) / 4

        extend_row = magnitude * np.sin(direction / 180 * np.pi)
        extend_col = magnitude * np.cos(direction / 180 * np.pi)

        row_image_last = reference_y - extend_row  # row and col not rounded yet
        col_image_last = reference_x - extend_col

        # for check_flow(), debug
        self.depth_x_now = depth_x
        self.depth_y_now = depth_y
        self.depth_z_now = depth_z
        self.row_image_now = reference_y
        self.col_image_now = reference_x
        self.mag_read = magnitude
        self.direc_read = direction

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
        depth_ul = depth_image_last[row_lidar_up, col_lidar_left]
        depth_ur = depth_image_last[row_lidar_up, col_lidar_right]
        depth_dl = depth_image_last[row_lidar_down, col_lidar_left]
        depth_dr = depth_image_last[row_lidar_down, col_lidar_right]

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

        depth_check1 = depth_image_last[row_lidar_up, col_lidar_left]
        depth_check2 = depth_image_last[row_lidar_up, col_lidar_right]
        depth_check3 = depth_image_last[row_lidar_down, col_lidar_left]
        depth_check4 = depth_image_last[row_lidar_down, col_lidar_right]

        index_good_points = np.where((depth_image_now < lidar_range) &  # current points inside lidar range
                                     # current points is not ground
                                     (ground_marker == 0) &
                                     # 4 raw vertices inside lidar FOV
                                     (row_lidar_up_raw >= 0) &
                                     (row_lidar_up_raw + 1 <= y_sample - 1) &
                                     (col_lidar_left_raw >= 0) &
                                     (col_lidar_left_raw + 1 <= x_sample - 1) &
                                     # 4 raw vertices located on the same surface of a object
                                     (obj_check1 > -1) & (obj_check2 > -1) & (obj_check3 > -1) & (obj_check4 > -1) &
                                     # 4 raw vertices inside lidar FOV
                                     (depth_check1 < lidar_range) & (depth_check2 < lidar_range) &
                                     (depth_check3 < lidar_range) & (depth_check4 < lidar_range))

        # take out those good points.
        depth_x_last = np.expand_dims(depth_x_last[index_good_points], axis=1)
        depth_y_last = np.expand_dims(depth_y_last[index_good_points], axis=1)
        depth_z_last = np.expand_dims(depth_z_last[index_good_points], axis=1)
        points_last = np.concatenate((depth_x_last, depth_y_last, depth_z_last), axis=1)  # refer to camera coord last

        depth_x = np.expand_dims(depth_x[index_good_points], axis=1)
        depth_y = np.expand_dims(depth_y[index_good_points], axis=1)
        depth_z = np.expand_dims(depth_z[index_good_points], axis=1)
        points_now = np.concatenate((depth_x, depth_y, depth_z), axis=1)  # refer to camera coord now
        # (camera coord move in world coord)
        # change the coord system time last to time now by using self movement
        # ie, points_last refer to camera coord now
        move_matrix = np.array([self.move_x, self.move_y, self.move_z])
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(self.delta_pitch, self.delta_roll,
                                                                           self.delta_yaw)

        points_last_new_coord = coords_ro_move(points_last, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                               mode='move_first')

        # speed relative to ground in current camera coord
        spd_xyz = (points_now - points_last_new_coord) / dt_in_game
        spd_mag = np.sqrt(np.sum(spd_xyz ** 2, axis=1))

        # build united sensitivity(of interpolation) matrix
        sensitivity_up = np.expand_dims(sensitivity_up, axis=2)
        sensitivity_down = np.expand_dims(sensitivity_down, axis=2)
        sensitivity_center = np.expand_dims(sensitivity_center, axis=2)
        sensitivity = np.concatenate((sensitivity_up, sensitivity_down, sensitivity_center), axis=2)

        # adaptive threshold,
        # m for detection moving points,
        # s for detection stationary points,
        # t for tracking those points from last points cloud
        threshold_m = depth_image_last * 0.05 + np.max(sensitivity, axis=2) * 0.95  # spd threshold
        threshold_s = threshold_m / 1 # spd threshold
        threshold_m_t = depth_image_last * 0.015 + np.max(sensitivity, axis=2) * 0.05  # position threshold
        threshold_s_t = depth_image_last * 0.015 + np.max(sensitivity, axis=2) * 0.05  # position threshold
        threshold_m_t_spd = 1 + depth_image_last * 0.05  # + np.max(sensitivity, axis=2) * 0.95  # spd threshold

        threshold_m = threshold_m[index_good_points]
        threshold_s = threshold_s[index_good_points]
        threshold_m_t = threshold_m_t[index_good_points]
        threshold_m_t_spd = threshold_m_t_spd[index_good_points]
        threshold_s_t = threshold_s_t[index_good_points]

        # take out the stationary points' now and last position refer to camera coord last
        stationary_index = np.where(spd_mag < threshold_s)
        s_points_now = points_now[stationary_index]
        s_points_last = points_last[stationary_index]
        threshold_s_t = threshold_s_t[stationary_index]
        lack = spd_mag - threshold_s
        t0 = time.time()
        # pairwise distance of each point in s_points_last to each point in last classified clouds (in camera coord last)
        spl_to_st = threshold_s_t + 1
        spl_to_sc2 = threshold_s_t + 1
        spl_to_sc = threshold_s_t + 1

        if s_tracked_coord_last.size !=0:
            spl_to_st = distance.cdist(s_points_last, s_tracked_coord_last, 'euclidean')
            # min. distance of each point in s_points_last to each point in last classified clouds
            spl_to_st = np.min(spl_to_st, axis=1)

        if s_candidate2_coord_last.size !=0:
            spl_to_sc2 = distance.cdist(s_points_last, s_candidate2_coord_last, 'euclidean')
            spl_to_sc2 = np.min(spl_to_sc2, axis=1)

        if s_candidate_coord_last.size !=0:
            spl_to_sc = distance.cdist(s_points_last, s_candidate_coord_last, 'euclidean')
            spl_to_sc = np.min(spl_to_sc, axis=1)

        # classify stationary points now (s_points_now) into 3 group
        # below, use bool index instead np.where for simplicity
        # s_tracked_index = np.where((spl_to_st < threshold_s_t) | (spl_to_sc2 < threshold_s_t))
        s_tracked_index = (spl_to_st < threshold_s_t) | (spl_to_sc2 < threshold_s_t)
        s_tracked_coord = s_points_now[s_tracked_index,:]

        #s_rest_index = np.where((spl_to_st >= threshold_s_t) & (spl_to_sc2 >= threshold_s_t))
        s_rest_index = ~s_tracked_index
        s_points_now_rest = s_points_now[s_rest_index,:]
        spl_to_sc_rest = spl_to_sc[s_rest_index]
        threshold_s_t = threshold_s_t[s_rest_index]

        # s_candidate2_index = np.where(spl_to_sc_rest < threshold_s_t)
        s_candidate2_index = (spl_to_sc_rest < threshold_s_t)
        s_candidate2_coord = s_points_now_rest[s_candidate2_index,:]

        #s_candidate_index = np.where(spl_to_sc_rest >= threshold_s_t)
        s_candidate_index = ~s_candidate2_index
        s_candidate_coord = s_points_now_rest[s_candidate_index,:]

        ## below for moving points
        # take out the moving points' now and last position refer to camera coord last
        moving_index = np.where(spd_mag > threshold_m)
        m_points_now = points_now[moving_index]
        m_spd_xyz = spd_xyz[moving_index]
        m_points_last = points_last[moving_index]
        threshold_m_t = threshold_m_t[moving_index]
        threshold_m_t_spd = threshold_m_t_spd[moving_index]
        excess = spd_mag - threshold_m

        # change current coord for last spd, so it can compare with current spd
        move_matrix = np.array([0,0,0])
        m_tracked_spd_last = coords_ro_move(m_tracked_spd_last, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                               mode='move_first')
        m_candidate2_spd_last = coords_ro_move(m_candidate2_spd_last, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                               mode='move_first')
        m_candidate_spd_last = coords_ro_move(m_candidate_spd_last, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                               mode='move_first')

        # pairwise distance of each point in m_points_last/m_spd_xyz to each point in last classified clouds (in camera coord last)
        # different than what for tracking stationary point, for moving points speed is also checked, pipe line below is a little different
        threshold_m_t = np.expand_dims(threshold_m_t, axis=1)
        threshold_m_t_spd = np.expand_dims(threshold_m_t_spd, axis=1)
        mpl_to_mt = threshold_m_t + 1
        mpl_to_mc2 = threshold_m_t + 1
        mpl_to_mc = threshold_m_t + 1
        mps_to_mt = threshold_m_t_spd + 1
        mps_to_mc2 = threshold_m_t_spd + 1
        mps_to_mc = threshold_m_t_spd + 1

        if m_tracked_coord_last.size !=0:
            mpl_to_mt = distance.cdist(m_points_last, m_tracked_coord_last, 'euclidean')
            mps_to_mt = distance.cdist(m_spd_xyz, m_tracked_spd_last, 'euclidean')

        if m_candidate2_coord_last.size !=0:
            mpl_to_mc2 = distance.cdist(m_points_last, m_candidate2_coord_last, 'euclidean')
            mps_to_mc2 = distance.cdist(m_spd_xyz, m_candidate2_spd_last, 'euclidean')

        if m_candidate_coord_last.size !=0:
            mpl_to_mc = distance.cdist(m_points_last, m_candidate_coord_last, 'euclidean')
            mps_to_mc = distance.cdist(m_spd_xyz, m_candidate_spd_last, 'euclidean')

        # classify stationary points now (s_points_now) into 3 group
        m_tracked_index1 = (mpl_to_mt < threshold_m_t) & (mps_to_mt < threshold_m_t_spd)
        m_tracked_index1 = np.max(m_tracked_index1, axis=1)
        m_tracked_index2 = (mpl_to_mc2 < threshold_m_t) & (mps_to_mc2 < threshold_m_t_spd)
        m_tracked_index2 = np.max(m_tracked_index2, axis=1)
        m_tracked_index = m_tracked_index1 | m_tracked_index2

        m_tracked_coord = m_points_now[m_tracked_index,:]
        m_tracked_spd = m_spd_xyz[m_tracked_index,:]

        m_rest_index = ~m_tracked_index
        m_points_now_rest = m_points_now[m_rest_index,:]
        m_spd_xyz_rest = m_spd_xyz[m_rest_index,:]
        mpl_to_mc_rest = mpl_to_mc[m_rest_index]
        mps_to_mc_rest = mps_to_mc[m_rest_index]
        threshold_m_t = threshold_m_t[m_rest_index,:]
        threshold_m_t_spd = threshold_m_t_spd[m_rest_index,:]

        m_candidate2_index = (mpl_to_mc_rest < threshold_m_t) & (mps_to_mc_rest < threshold_m_t_spd)
        m_candidate2_index = np.max(m_candidate2_index, axis=1)
        m_candidate2_coord = m_points_now_rest[m_candidate2_index,:]
        m_candidate2_spd = m_spd_xyz_rest[m_candidate2_index,:]

        m_candidate_index = ~m_candidate2_index
        m_candidate_coord = m_points_now_rest[m_candidate_index,:]
        m_candidate_spd = m_spd_xyz_rest[m_candidate_index,:]
        print('tracking takes', time.time() -t0)
        # add markers to show image
        marker = None
        if self.plot != 0:
            row = np.expand_dims(reference_y[index_good_points], axis=1)
            col = np.expand_dims(reference_x[index_good_points], axis=1)
            reference = np.concatenate((col, row), axis=1)

            s_reference = reference[stationary_index]
            marker_s_tracked = s_reference[s_tracked_index]
            s_reference_rest = s_reference[s_rest_index]

            marker_s_candidate2 = s_reference_rest[s_candidate2_index]
            marker_s_candidate = s_reference_rest[s_candidate_index]

            m_reference = reference[moving_index]
            marker_m_tracked = m_reference[m_tracked_index]
            m_reference_rest = m_reference[m_rest_index]

            marker_m_candidate2 = m_reference_rest[m_candidate2_index]
            marker_m_candidate = m_reference_rest[m_candidate_index]

            marker = {'marker_tracked': np.array(marker_m_tracked),
                      'marker_candidate2': np.array(marker_m_candidate2),
                      'marker_candidate': np.array(marker_m_candidate),
                      'marker_s_tracked': np.array(marker_s_tracked),
                      'marker_s_candidate2': np.array(marker_s_candidate2),
                      'marker_s_candidate': np.array(marker_s_candidate)}

        esti_coord = None
        # add more data
        if self.plot == 2:
            s_tracked_coord_e = s_points_last[s_tracked_index]
            m_tracked_coord_e = m_points_last[m_tracked_index]

            move_matrix = np.array([-self.move_x, -self.move_y, -self.move_z])
            rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(-self.delta_pitch, -self.delta_roll,-self.delta_yaw)
            points_last0 = coords_ro_move(points_now, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='turn_first')

            s_points_last0 = points_last0[np.where(stationary_index)]
            m_points_last0 = points_last0[np.where(moving_index)]

            s_tracked_coord_e0 = s_points_last0[s_tracked_index]
            m_tracked_coord_e0 = m_points_last0[m_tracked_index]

            esti_coord = {'tracked_coord_e': np.array(m_tracked_coord_e),
                          'tracked_coord_e0': np.array(m_tracked_coord_e0),
                          's_tracked_coord_e': np.array(s_tracked_coord_e),
                          's_tracked_coord_e0': np.array(s_tracked_coord_e0)}

        self.m_tracked_coord_last = m_tracked_coord
        self.m_tracked_spd_last = m_tracked_spd
        self.m_candidate_coord_last = m_candidate_coord
        self.m_candidate_spd_last = m_candidate_spd
        self.m_candidate2_coord_last = m_candidate2_coord
        self.m_candidate2_spd_last = m_candidate2_spd

        self.s_tracked_coord_last = s_tracked_coord
        self.s_candidate_coord_last = s_candidate_coord
        self.s_candidate2_coord_last = s_candidate2_coord


        self.depth_image_last = self.depth_image_now

        # return the current data actually
        return m_tracked_spd, self.m_tracked_coord_last, self.s_tracked_coord_last, excess, lack, \
               reference_y, reference_x, marker, esti_coord



    # debug function
    def check_flow(self):
        move_matrix = np.array([-self.move_x, -self.move_y, -self.move_z])
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(-self.delta_pitch, -self.delta_roll,
                                                                           -self.delta_yaw)

        depth_x = np.expand_dims(self.depth_x_now, axis=2)
        depth_y = np.expand_dims(self.depth_y_now, axis=2)
        depth_z = np.expand_dims(self.depth_z_now, axis=2)
        all_point_now = np.concatenate((depth_x,depth_y,depth_z), axis=2).reshape(-1,3)
        all_points_last_esti = coords_ro_move(all_point_now, move_matrix, rotation_matrix_p_r, rotation_matrix_y,
                                              mode='turn_first')

        depth_x_last = all_points_last_esti[:,0].reshape(self.y_sample, self.x_sample)
        depth_y_last = all_points_last_esti[:,1].reshape(self.y_sample, self.x_sample)
        depth_z_last = all_points_last_esti[:,2].reshape(self.y_sample, self.x_sample)
        # cartesian to image coord
        row_image_last0 = (self.v_sample - 1) / 2 - depth_y_last * self.focus / depth_z_last
        col_image_last0 = (self.h_sample - 1) / 2 - depth_x_last * self.focus / depth_z_last

        flow_y = self.row_image_now - row_image_last0
        flow_x = self.col_image_now - col_image_last0

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




