
import time
import autograd.numpy as np
#import numpy as np

from autograd import grad, hessian
from scipy.optimize import minimize, Bounds

from Lidar_tools_AG import build_rotation_matrix_2D, coords_ro_move_2D, coords_ro_move, build_roatation_matrix_3D#, coords_ro_move2



def rho_theta(line):
    if line[3] - line[1] !=0:
        theta = np.arctan((line[0] - line[2]) / (line[3] - line[1]))
    else:
        theta = np.pi/2
    rho = line[0] * np.cos(theta) + line[1] * np.sin(theta)

    return rho, theta


def state_update(target_lane, tracked_coord_last, speed_tracked, s_tracked_coord_last, v_last, dt, gas_brake, steer, launch_factor,
                 drive_factor, brake_factor, slope_factor, drag_factor_constant, drag_factor_linear, drag_factor_air):
    # acc = k*u/v + s + w0 + w1*v + w2*v**2
    if v_last > 0:
        if gas_brake > 0:
            acc = drive_factor * gas_brake / v_last + slope_factor + \
                  drag_factor_constant + drag_factor_linear * v_last + drag_factor_air * v_last ** 2
        else:
            acc = brake_factor * gas_brake + slope_factor + \
                  drag_factor_constant + drag_factor_linear * v_last + drag_factor_air * v_last ** 2
    else:
        acc = gas_brake * launch_factor + drag_factor_constant

    # drive backwards not modeled


    correct_factor = 1 - acc * 0.01
    delta_yaw = v_last * dt * np.sin(steer) / 1.5 * correct_factor



    #if type(a) is not np.numpy_boxes.ArrayBox:
    #    print('+++ ',brake_factor, drive_factor, a, gas_brake, v_last)

    ### update vehcle state in the future
    if delta_yaw != 0:
        horizontal_turn_radius = (v_last * dt + acc * dt ** 2 / 2) / delta_yaw
        move_z = horizontal_turn_radius * np.sin(delta_yaw)
        move_x = horizontal_turn_radius * (1 - np.cos(delta_yaw))

    else:
        move_z = v_last * dt + acc * dt ** 2 / 2
        move_x = 0

    v_new = v_last + acc * dt

    ### environment_state_update in the future
    # first, move self moving points
    tracked_coord_last = tracked_coord_last + speed_tracked * dt

    # vehicle move and rotate
    trans_matrix = np.array([move_x, 0, move_z])
    rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(0, 0, delta_yaw)
    tracked_coord_new = coords_ro_move(tracked_coord_last, trans_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='move_first')
    s_tracked_coord_new = coords_ro_move(s_tracked_coord_last, trans_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='move_first')

    #tracked_coord_new = coords_ro_move2(tracked_coord_last, move_x, 0, move_z, 0, 0, delta_yaw,
    #                                    mode='move_first')
    #s_tracked_coord_new = coords_ro_move2(s_tracked_coord_last, move_x, 0, move_z, 0, 0, delta_yaw,
    #                                      mode='move_first')

    # width, height, distance condition
    if tracked_coord_new.size !=0:
        if s_tracked_coord_new.size !=0:
            all_coord = np.vstack((tracked_coord_new, s_tracked_coord_new))
        else:
            all_coord = tracked_coord_new
    else:
        if s_tracked_coord_new.size !=0:
            all_coord = s_tracked_coord_new
        else:
            all_coord = np.array([])


    distance_z = None
    distance_x = None
    if all_coord.size != 0:
        index_in_range = np.where((np.abs(all_coord[:, 0]) < 1.2) & (all_coord[:, 1] > -10) & (all_coord[:, 1] < 2))[0]
        if index_in_range.size != 0:
            filtered_coord = all_coord[index_in_range, :]  # [0]
            index_nearst = np.argmin(filtered_coord[:, 2])
            # print(filtered_coord)
            # distance = np.min(filtered_coord[:,2])
            # distance = filtered_coord[index_nearst, 2] + np.abs(np.mean(filtered_coord[:, 0]))
            # distance = np.sqrt(filtered_coord[index_nearst, 2]**2 + (filtered_coord[index_nearst, 0] * 2)**2)
            distance_x = filtered_coord[index_nearst, 0]
            distance_z = filtered_coord[index_nearst, 2]


    ### update target lane
    move_matrix = np.array([move_x, move_z])
    roatation_matrix = build_rotation_matrix_2D(delta_yaw)
    target_lane_new = np.hstack((coords_ro_move_2D(target_lane[0:2], move_matrix, roatation_matrix, mode='move first'),
                                 coords_ro_move_2D(target_lane[2:4], move_matrix, roatation_matrix, mode='move first')))


    #distance = np.inf
    #tracked_coord_new = tracked_coord_last
    #s_tracked_coord_new = s_tracked_coord_last
    return target_lane_new, tracked_coord_new, s_tracked_coord_new, v_new, distance_x, distance_z


class MPC:
    def __init__(self):
        self.tracked_coord = None # self moving points
        self.speed_tracked = None # speed of self moving points
        self.s_tracked_coord = None # stationary points
        self.pitch = None # vehicle pitch
        self.roll = None
        self.yaw_rate = None
        self.steering_angle = None
        self.gas_brake= None
        self.speed = None
        self.acc = None
        self.target_lane = None

        #self.steer_factor_low_v = None
        self.brake_factor = 22
        self.steer_factor = None
        self.launch_factor = 7.3
        self.drive_factor = 170
        self.slope_factor = 0
        self.drag_factor_constant = -3
        self.drag_factor_linear = -0.009
        self.drag_factor_air = -0.001


        self.initialized = 0
        self.count_no_lane = 0
        self.x_last_esti = None

        self.latency = 0.25 # seconds
        self.keep_distance = 15 # meter, distance to front object
        self.detection_range = 30 # meter
        self.target_speed = 7 # m/s
        self.half_avoid_width = 1 # inside this width, the distance to front objects will be keeped

    def input(self, tracked_coord, speed_tracked, s_tracked_coord, pitch, roll, yaw_rate, steering_angle, gas_brake, speed, acc, target_lane):
        self.tracked_coord = tracked_coord  # self moving points, in camera coord
        self.speed_tracked = speed_tracked
        self.s_tracked_coord = s_tracked_coord  # stationary points, in camera coord
        self.pitch = pitch  # vehicle pitch
        self.roll = roll
        self.yaw_rate = yaw_rate
        self.steering_angle = steering_angle
        self.gas_brake = gas_brake
        self.speed = speed
        self.acc = acc
        self.target_lane = target_lane


    def error_function(self, control_inputs):
        #steer_factor_low_v = self.steer_factor_low_v
        steer_factor = self.steer_factor
        launch_factor = self.launch_factor
        drive_factor = self.drive_factor
        brake_factor = self.brake_factor
        slope_factor = self.slope_factor
        drag_factor_constant = self.drag_factor_constant
        drag_factor_air = self.drag_factor_air
        drag_factor_linear = self.drag_factor_linear
        tracked_coord = self.tracked_coord
        speed_tracked = self.speed_tracked
        s_tracked_coord = self.s_tracked_coord
        target_line = self.target_lane
        target_speed = self.target_speed
        keep_distance = self.keep_distance
        detection_range = self.detection_range
        v = self.speed
        dt = self.latency

        gas_brake_last = self.gas_brake
        steer_angle_last = self.steering_angle

        # each a and delta_yaw are in X
        error_cross_track = 0
        error_phi = 0
        error_keep_distance = 0
        error_speed = 0
        pena_gas = 0
        pena_steer = 0

        for n in range(0, control_inputs.size, 2):
            gas_brake = control_inputs[n]
            steer = control_inputs[n + 1]
            target_lane_new, tracked_coord_new, s_tracked_coord_new, v_new, distance_x, distance_z = \
                state_update(target_line, tracked_coord, speed_tracked, s_tracked_coord, v, dt, gas_brake, steer, launch_factor,
                              drive_factor, brake_factor, slope_factor, drag_factor_constant, drag_factor_linear, drag_factor_air)

            rho, theta = rho_theta(target_lane_new)
            # accumulate errors
            error_cross_track += rho ** 2
            error_phi += theta ** 2 *10
            error_speed += (target_speed - v) ** 2

            #if distance < detection_range:
            #    #ed = (1 / np.exp(distance - keep_distance) - 1 / np.exp(detection_range- keep_distance)) * 500
            #    ed = np.exp(keep_distance - distance) * v *10
            #    error_keep_distance += ed
            #else:
            #    ed = 0

            if distance_z is not None:
                ed = np.max(np.exp(10 - distance_z + 10 * (1 - np.abs(distance_x)))) * np.abs(v)
            else:
                ed = 0


            error_keep_distance += ed


            # penalization
            pena_gas += (gas_brake_last - gas_brake) ** 2 *400
            pena_steer += (steer_angle_last - steer) ** 2 *400

            if type(rho) is not np.numpy_boxes.ArrayBox:
                print('*** some errors:  ',(target_speed - v) ** 2, theta ** 2 *10, ed)


            # set for next loop
            gas_brake_last = gas_brake
            steer_angle_last = steer
            tracked_coord, s_tracked_coord, v, target_line = tracked_coord_new, s_tracked_coord_new, v_new, target_lane_new

            #print('each error sum: ', error_cross_track, error_phi, error_speed, error_keep_distance, pena_gas, pena_steer)
        target_function = error_cross_track + error_phi + error_speed + error_keep_distance + pena_gas + pena_steer

        return target_function

    def update_steer_factor(self): # currently not in use
        steering_angle = self.steering_angle
        speed = self.speed
        yaw_rate = self.yaw_rate
        latency = self.latency

        if steering_angle !=0 and (yaw_rate / steering_angle) >0:
            self.steer_factor = abs(yaw_rate / steering_angle * speed)
        ## simple approximate factors
        #if abs(steering_angle) > 0.1 and abs(speed) > 5 and abs(yaw_rate) > 0.1:
        #    self.steer_factor = abs(yaw_rate / steering_angle * speed)  # constant ratio (slip angle(force) / steering angle)
        #else:
        #    self.steer_factor = abs(steering_angle * 30 / 180 * np.pi / latency)  # default, no slip angle
        print('------ sf: ', self.steer_factor)#


    def update_slope_factor(self):
        actual_acc = self.acc
        gas_brake = self.gas_brake
        speed = self.speed
        drive_factor = self.drive_factor
        drag_factor_constant = self.drag_factor_constant
        drag_factor_air = self.drag_factor_air
        drag_factor_linear = self.drag_factor_linear
        brake_factor = self.brake_factor

        # acc = k*u/v + s + w0 + w1*v + w2*v**2
        # s = acc - (k*u/v + w0 + w1*v + w2*v**2)
        # (for brake term k*u/v is k*u)
        if speed > 0:
            if gas_brake >= 0:
                slope_factor = actual_acc - (drive_factor * gas_brake / speed +
                    drag_factor_constant + drag_factor_linear * speed + drag_factor_air * speed ** 2)
            else:
                slope_factor = actual_acc - (brake_factor * gas_brake +
                    drag_factor_constant + drag_factor_linear * speed + drag_factor_air * speed ** 2)

            # mean from previous and current value
            self.slope_factor += slope_factor
            self.slope_factor *= 0.5


    def run(self):
        tracked_coord = self.tracked_coord  # self moving points, in camera coord
        speed_tracked = self.speed_tracked
        s_tracked_coord = self.s_tracked_coord  # stationary points, in camera coord
        pitch = self.pitch  # vehicle pitch
        roll = self.roll
        yaw_rate = self.yaw_rate
        steering_angle = self.steering_angle
        gas_brake = self.gas_brake
        speed = self.speed
        acc = self.acc
        target_lane = self.target_lane
        latency = self.latency
        x_last_esti = self.x_last_esti

        if target_lane is not None:
            self.initialized = 1
        else:
            if self.initialized == 0:
                return [0, 0, 0]
            else:
                command = [0, 0, -x_last_esti[3]]
                return command


        # remove vehicle pitch and roll
        move_matrix = [0,0,0]
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(-pitch, -roll, 0)
        tracked_coord = coords_ro_move(tracked_coord, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='turn_first')
        s_tracked_coord = coords_ro_move(s_tracked_coord, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='turn_first')
        self.speed_tracked = coords_ro_move(speed_tracked, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='turn_first')

        # calculate initial state after latency
        self.speed += acc * latency
        delta_yaw = yaw_rate * latency
        if delta_yaw != 0:
            #  simple ver. only using the yaw to estimate, in camera coord: x,y,z refer to front, width, height
            horizontal_turn_radius = (speed * latency + acc * latency ** 2 / 2) / delta_yaw
            move_z = horizontal_turn_radius * np.sin(delta_yaw)
            move_x = horizontal_turn_radius * (1 - np.cos(delta_yaw))
        else:
            move_z = (speed * latency + acc * latency ** 2 / 2)
            move_x = 0

        # pre calculate the state at the time all calculation is done
        move_matrix = [move_x, 0, move_z]
        rotation_matrix_p_r, rotation_matrix_y = build_roatation_matrix_3D(0, 0, delta_yaw)
        self.tracked_coord = coords_ro_move(tracked_coord, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='move_first')
        self.s_tracked_coord = coords_ro_move(s_tracked_coord, move_matrix, rotation_matrix_p_r, rotation_matrix_y, mode='move_first')

        move_matrix = np.array([move_x, move_z])
        rotation_matrix = build_rotation_matrix_2D(delta_yaw)
        p1 = coords_ro_move_2D(target_lane[0:2], move_matrix, rotation_matrix, mode='move first')
        p2 = coords_ro_move_2D(target_lane[2:4], move_matrix, rotation_matrix, mode='move first')
        self.target_lane = np.hstack((p1, p2))


        ### MPC

        self.update_steer_factor()
        self.update_slope_factor()

        if x_last_esti is not None:
            x0 = x_last_esti[2]
            x1 = x_last_esti[3]
        else:
            x0 = 0.7
            x1 = 0.0

        control_input0 = np.array([x0,x1,x0,x1,x0,x1,x0,x1])
        lb = np.array([-1, -1, -1, -1, -1, -1, -1, -1]) *0.5
        ub = np.array([1, 1, 1, 1, 1, 1, 1, 1]) *0.5

        jacobian_matrix = grad(self.error_function)
        hessian_matrix = hessian(self.error_function)
        bounds = Bounds(lb, ub)

        t0 = time.time()
        res = minimize(self.error_function, control_input0, method='trust-constr', jac=jacobian_matrix, hess=hessian_matrix,
                       bounds=bounds, tol=0.1)
                       #options = {'verbose': 1}, bounds = bounds)
        print('solve minimization takes: ', time.time() - t0)
        print('Command serie: ', res.x)


        ### update and output result

        self.x_last_esti = res.x

        control_gas_brake = res.x[0]
        if control_gas_brake < 0:
            control_brake = -control_gas_brake
            control_gas = 0
        else:
            control_brake = 0
            control_gas = control_gas_brake
        control_steer = res.x[1] # here left is +

        return [control_gas, control_brake, control_steer]








