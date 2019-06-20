#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from deepgtav.messages import Start, Stop, Scenario, Dataset, Commands, frame2numpy
from deepgtav.client import Client

import argparse
import time
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import math

import image_preprocess as IMG
import Ground_esti as GE
#import Movement_check3 as MC3c  # point to cloud
import Movement_check4 as MC3c  # point to point
import Lane_finder as LF
import Lane_controler as LC
import Vehicle_control as Vc



def avoid_dead_zone(x):
    if x>0:
        return x * 0.75 + 0.25
    elif x<0:
        return x * 0.75 - 0.25
    else:
        return 0


# Controls the DeepGTAV vehicle
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('-l', '--host', default='localhost', help='The IP where DeepGTAV is running')
    parser.add_argument('-p', '--port', default=8000, help='The port where DeepGTAV is running')
    args = parser.parse_args()

    # Creates a new connection to DeepGTAV using the specified ip and port. 
    # If desired, a dataset path and compression level can be set to store in memory all the data received in a gziped pickle file.
    # We don't want to save a dataset in this case
    client = Client(ip=args.host, port=args.port)

    # We set the scenario to be in manual driving, and everything else random (time, weather and location). 
    # See deepgtav/messages.py to see what options are supported
    #scenario = Scenario(drivingMode=-1) #manual driving
    Route0 = [-1989.000000, -468.250000, 10.562500,
             1171.351563, -1925.791748, 36.220097]

    Route1 = [-1889.000000, -368.250000, 10.562500,
             1171.351563, -1925.791748, 36.220097]

    Route2 = [-1889.000000, -388.250000, 10.562500,
              1171.351563, -1925.791748, 36.220097]

    Route3 = [-1689.000000, 188.250000, 10.562500,
              1171.351563, -1925.791748, 36.220097]

    # time_scale, time_factor:
    # 0.1         10
    # 0.05        20
    # 0.02        50
    time_factor = 20 # real time to 1s in game (also change latency in MPC)
    scenario = Scenario(time_scale=1/time_factor, weather='EXTRASUNNY', vehicle='blista', time=[12, 0], drivingMode=[-2, 0, 5.0, 1.0, 1.0],
                        route=Route0)
    #scenario = Scenario(time_scale=0.1, weather='EXTRASUNNY', vehicle='blista', time=[12, 0],
     #                   drivingMode=[0, 0, 3.0, 1.0, 1.0])


    lidar_range = 100
    lidar_X_res = 160
    lidar_Y_res = 40
    lidar_L_FOV = 40
    lidar_R_FOV = 320
    lidar_U_FOV = 85
    lidar_D_FOV = 100
    lidar_para = [2, False, lidar_range, lidar_X_res, lidar_L_FOV, lidar_R_FOV, lidar_Y_res, lidar_U_FOV, lidar_D_FOV]

    dataset = Dataset(rate=None, frame=[1280, 720], throttle=True, brake=True, steering=True, speed=True,
                      acceleration=True, yaw=True, pitch=True, roll=True, yawRate=True, pitchRate=True, rollRate=True, isCollide=False, location=False,
                      drivingModeMsg=False, time=False,
                      lidar=lidar_para)  # , vehicles=True, peds=True)

    # Send the Start request to DeepGTAV. Dataset is set as default, we only receive frames at 10Hz (320, 160)
    client.sendMessage(Start(scenario=scenario, dataset=dataset))


    # parameters
    Camera_height_FOV = 60
    focus = 719 / 2 / math.tan(Camera_height_FOV / (180 / np.pi) / 2)  # the FOV in GTAV is refer to the shorter dimension of the picture
    LiDAR_FOV_x = lidar_L_FOV + 360 - lidar_R_FOV
    LiDAR_FOV_y = lidar_D_FOV - lidar_U_FOV

    x_sample_deg = LiDAR_FOV_x / (lidar_X_res-1)
    y_sample_deg = LiDAR_FOV_y / (lidar_Y_res-1)

    Lidar_info = {'x_sample': lidar_X_res,
                  'x_sample_deg': x_sample_deg,
                  'y_sample': lidar_Y_res,
                  'y_sample_deg': y_sample_deg,
                  'upper_lim': lidar_U_FOV,
                  'lidar_range': lidar_range}

    Camera_info1 = {'v_sample':300,
                   'h_sample':1100,
                    'focus':focus}

    Camera_info2 = {'v_sample': 450,
                    'h_sample': 1100,
                    'focus': focus}
    Opt_flow = IMG.Optical_flow()
    Ground_checker = GE.Ground_esti(Lidar_info)
    Movement_checker = MC3c.Move_esti(Lidar_info, Camera_info1)
    lane_controler = LC.lane_controler()
    Lane_finder = LF.lane_finder(Lidar_info,Camera_info2)
    vehicle_controler = Vc.MPC()

    stoptime = time.time() + 1*3600
    count = 0
    time_last = None  # initial time
    while time.time() < stoptime:
        try:
            if count == 0:
                time.sleep(10)

            count += 1
            time_now = time.time()  # time at receiving message
            # We receive a message as a Python dictionary
            client.recvMessage()
            client.recvMessage()
            message = client.recvMessage()
            print('receive data takes: ', time.time() - time_now)
            # Note: lazy method to solve bug from deep(er)GTAV: a recv call at n+1 will give you the data from call at n,
            # so just call it 2 times, but only use the data from the last time. one recv call takes about 0.2~0.3s.
            # lidar data seems without this problem?
            lidar_img = message['lidar']
            lidar_img.shape = (lidar_Y_res, lidar_X_res)
            lidar_img = np.flip(lidar_img, axis=0)

            image_raw = frame2numpy(message['frame'], (1280, 720))
            # 3 place to change: here, IMG:coord move, MC:check_flow, ...
            image_2 = image_raw[(360 - 150): (360 + 300), (640 - 550):(640 + 550), :]
            image_2 = cv2.cvtColor(image_2, cv2.COLOR_BGR2RGB)

            image_1 = cv2.cvtColor(image_raw, cv2.COLOR_BGR2GRAY)
            image_1 = image_1[(360 - 150): (360 + 150), (640 - 550): (640 + 550)]
            #image_1 = cv2.resize(image_1, (825, 225))
            #image_1 = cv2.resize(image_1, (550, 150))


            # read more message
            speed_now = message['speed']
            Yaw_now = message['yaw']
            yawRate_ingame = message['yawRate']
            pitch = - message['pitch']
            roll = - message['roll'] # transfer to camera coord --- right is +
            gas_brake = message['throttle'] - message['brake']
            steering_angle = - message['steering'] / 0.9 # normalize and change to camera coord left is +

            # lidar to image reference
            #t0 = time.time()
            #reference_x, reference_y = IMG.coord_trans(lidar_img, Lidar_info, Camera_info1)  # output is float
            #print('reference takes: ', time.time()-t0)

            # optical flow
            Opt_flow.input(image_1)
            magnitude, direction, image_flow = Opt_flow.run()

            #if magnitude is not None:
            #    magnitude = cv2.resize(magnitude, (1100, 300))
            #    direction = cv2.resize(direction, (1100, 300), interpolation=cv2.INTER_NEAREST)
            #    image_flow = cv2.resize(image_flow, (1100, 300))
            #    #magnitude2 = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')
            #    cv2.imshow("image_flow", image_flow)
            #    cv2.waitKey(2)

            t0 = time.time()
            Ground_checker.input(lidar_img)
            ground_marker, pitch_lidar, roll_lidar = Ground_checker.run()
            print('ground checker takes: ', time.time()-t0)

            # main calculation
            if count > 1:

                dt_in_game = (time_now - time_last) / time_factor

                # yaw, pitch, roll
                if Yaw_now < 0 and Yaw_last > 0 and yawRate_ingame > 0:
                    delta_yaw = np.pi + Yaw_now + np.pi - Yaw_last
                elif Yaw_now > 0 and Yaw_last < 0 and yawRate_ingame < 0:
                    delta_yaw = - (np.pi - Yaw_now + np.pi + Yaw_last)
                else:
                    delta_yaw = Yaw_now - Yaw_last

                yaw_rate = delta_yaw / dt_in_game
                acc = (speed_now - speed_last) / dt_in_game
                delta_pitch = pitch - pitch_last
                delta_roll = roll - roll_last

                t0 = time.time()
                Movement_checker.input(ground_marker, magnitude, direction, lidar_img, dt_in_game,
                                        pitch_lidar, delta_roll, delta_pitch, delta_yaw, speed_now, acc)

                speed_tracked, tracked_coord, s_tracked_coord, over_shoot, down_shoot, reference_y, reference_x, \
                marker, esti_coord = Movement_checker.run()
                print('movement checker takes: ', time.time() - t0)

                t0 = time.time()
                Lane_finder.input(reference_x, reference_y, ground_marker, lidar_img, image_2,
                                  delta_yaw, roll_lidar, pitch_lidar, acc, speed_now, dt_in_game)

                lanes, map_image, move_x, move_z, lane_marker_index_w, lane_marker_index_y, \
                image_y_balanced, image_gray_balanced, image_warped = Lane_finder.run()
                print('lane finder takes: ', time.time() - t0)

                t0 = time.time()
                lane_controler.input(lanes, map_image, delta_yaw, move_x, move_z, speed_now, acc, delta_yaw, dt_in_game) #########
                image_lane_control, target_lane = lane_controler.run()
                print('lane controler takes: ', time.time() - t0)


                t0 = time.time()
                vehicle_controler.input(tracked_coord, speed_tracked, s_tracked_coord, pitch_lidar, roll_lidar,
                                        yaw_rate, steering_angle, gas_brake, speed_now, acc, target_lane)
                command = vehicle_controler.run()
                print('sttering:  ', steering_angle)
                ##print('Command: ', command) # in camera coord, turn left is +
                ##print('Comannd now: ', steering_angle)
                print('vehicle controler takes: ', time.time() - t0)

                client.sendMessage(Commands(avoid_dead_zone(command[0]), avoid_dead_zone(command[1]), - avoid_dead_zone(command[2]))) # steering is opposite for GTAV
                #client.sendMessage(Commands(0.8, 0, 0.3))

                all_calculation_time = time.time() - time_now

                # monitor
                if reference_x is not None:
                    if 1: # for movement detectoin
                        marker_ground_ref = GE.ground_marker_2_image(ground_marker, reference_x, reference_y)
                        image_monitor = IMG.lidar_result_fusion(image_2, marker, marker_ground_ref)

                        #if lanes_3d is not None:
                        #    image_monitor = IMG.test_show_lidar_line(image_monitor, lanes_3d, focus)

                        #image_monitor = IMG.lane_result_fusion(image_monitor, lines_original, lanes_original)

                        font = cv2.FONT_HERSHEY_SIMPLEX
                        text = 'command: {}, {}, {}'.format(round(command[0], 3), round(command[1], 3), round(command[2], 3))
                        cv2.putText(image_monitor, text, (10, 30), font, 1, (0, 0, 100), 1, cv2.LINE_AA)
                        text = 'speed: {}'.format(round(speed_now,2))
                        cv2.putText(image_monitor, text, (10, 55), font, 1, (0, 0, 100), 1, cv2.LINE_AA)
                        text = 'dt: {}'.format(round(dt_in_game * time_factor, 2))
                        cv2.putText(image_monitor, text, (10, 80), font, 1, (0, 0, 100), 1, cv2.LINE_AA)

                        cv2.imshow('monitor', image_monitor)
                        cv2.waitKey(2)


                    if 1: # lanes

                        image_lane_control = cv2.cvtColor(image_lane_control, cv2.COLOR_RGB2BGR)
                        cv2.imshow('map_image', image_lane_control)
                        cv2.waitKey(2)

                        #combined_img = np.concatenate((image_y_balanced, lane_marker_index_y), axis=1)
                        #cv2.imshow('lane_marker y', combined_img)
                        #cv2.waitKey(2)

                        #image_gray_balanced = cv2.cvtColor(image_gray_balanced, cv2.COLOR_GRAY2BGR)
                        #combined_img = np.concatenate((image_gray_balanced, lane_marker_index_w), axis=1)
                        #cv2.imshow('lane_marker w', combined_img)
                        #cv2.waitKey(2)

                        #image_y_balanced = cv2.cvtColor(image_y_balanced, cv2.COLOR_GRAY2BGR)
                        #image_gray_balanced = cv2.cvtColor(image_gray_balanced, cv2.COLOR_GRAY2BGR)
                        #image_warped = cv2.cvtColor(image_warped, cv2.COLOR_RGB2BGR)
                        #combined_img = np.concatenate((image_y_balanced, image_warped, image_gray_balanced), axis=1)
                        #cv2.imshow('image_balanced', combined_img)
                        #cv2.waitKey(2)

                    if 0:  # check estimate optical flow
                        image_flow_cut, image_flow_cut_0, mag_read, mag_esti, direc_read, direc_esti = Movement_checker.check_flow()
                        combine_flow = np.concatenate((image_flow_cut_0, image_flow_cut), axis=0)
                        cv2.imshow('monitor flow2',cv2.cvtColor(combine_flow, cv2.COLOR_RGB2BGR))
                        cv2.waitKey(2)

                print('all calculation takes ', all_calculation_time, ' s')
                print('incl. monitor takes ', time.time() - time_now, ' s')
                print(' ')


                if count > 8000 or (cv2.waitKey(20) & 0xFF == ord('q')):
                    #### all things below are for debug for point to cloud tracking only
                    #image_flow_cut, image_flow_cut_0, mag_read, mag_esti, direc_read, direc_esti = MC3u2.flow_esti(magnitude, direction, lidar_img_now, lidar_img_last, image_dt, focus, result_x, result_y,lidar_U_FOV, LiDAR_FOV_x, LiDAR_FOV_y, lidar_range, speed_last, speed_now, yawRate)

                    image_flow_cut, image_flow_cut_0, mag_read, mag_esti, direc_read, direc_esti = Movement_checker.check_flow()

                    # PLOT FIGURE ######################################################################################
                    if 1:
                        marker_tracked = marker['marker_tracked']
                        marker_candidate2 = marker['marker_candidate2']
                        marker_candidate = marker['marker_candidate']

                        plt.figure()
                        plt.subplot(3,1,1)
                        plt.imshow(image_1, cmap='gray')
                        #plt.plot(reference_x, reference_y, 'bo', markersize=0.2)
                        if marker_tracked.size != 0:
                            plt.plot(marker_tracked[:,0], marker_tracked[:,1], 'rx', markersize=1)
                        if marker_candidate2.size != 0:
                            plt.plot(marker_candidate2[:,0], marker_candidate2[:,1], 'yx', markersize=1)
                        if marker_candidate.size != 0:
                            plt.plot(marker_candidate[:,0], marker_candidate[:,1], 'gx', markersize=1)
                        plt.plot(marker_ground_ref[:, 0], marker_ground_ref[:, 1], 'cx', markersize=2)

                        plt.subplot(3, 1, 2)
                        plt.imshow(image_flow_cut_0)

                        plt.subplot(3, 1, 3)
                        plt.imshow(image_flow_cut)

                    if 0:
                        fig = plt.figure('1') # points position cloud
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(s_tracked_coord[:, 0], s_tracked_coord[:, 2], s_tracked_coord[:, 1], c='r', marker='.')
                        ax.scatter(s_tracked_coord_e[:, 0], s_tracked_coord_e[:, 2], s_tracked_coord_e[:, 1], c='b', marker='.')
                        ax.scatter(s_tracked_coord_e0[:, 0], s_tracked_coord_e0[:, 2], s_tracked_coord_e0[:, 1], c='g', marker='.')
                        ax.set_xlabel('width')
                        ax.set_xlim(-50, 50)
                        #ax.set_xlim(-5, 5)
                        ax.set_ylabel('depth')
                        ax.set_ylim(0, 110)
                        #ax.set_ylim(40, 45)
                        ax.set_zlabel('height')
                        ax.set_zlim(-50, 50)
                        #ax.set_zlim(-2, -1)
                    elif 1:
                        tracked_coord_e = esti_coord['tracked_coord_e']
                        tracked_coord_e0 = esti_coord['tracked_coord_e0']
                        fig = plt.figure('2')  # points position cloud
                        ax = fig.add_subplot(111, projection='3d')
                        ax.scatter(tracked_coord[:, 0], tracked_coord[:, 2], tracked_coord[:, 1], c='r',marker='.')
                        ax.scatter(tracked_coord_e[:, 0], tracked_coord_e[:, 2], tracked_coord_e[:, 1], c='b',marker='.')
                        ax.scatter(tracked_coord_e0[:, 0], tracked_coord_e0[:, 2], tracked_coord_e0[:, 1], c='g',marker='.')
                        ax.set_xlabel('width')
                        ax.set_xlim(-50, 50)
                        # ax.set_xlim(-5, 5)
                        ax.set_ylabel('depth')
                        ax.set_ylim(0, 110)
                        # ax.set_ylim(40, 45)
                        ax.set_zlabel('height')
                        ax.set_zlim(-50, 50)
                        # ax.set_zlim(-2, -1)

                    else:
                        fig = plt.figure() # points vector field
                        ax = fig.gca(projection='3d')
                        if tracked_coord.size != 0:
                            ax.quiver(tracked_coord[:, 0], tracked_coord[:, 2], tracked_coord[:, 1],
                                      np.zeros_like(over_shoot),np.zeros_like(over_shoot), over_shoot, length=0.05, color='g')
                        if s_tracked_coord.size != 0:
                            ax.quiver(s_tracked_coord[:, 0], s_tracked_coord[:, 2], s_tracked_coord[:, 1], np.zeros_like(down_shoot),
                                      np.zeros_like(down_shoot), down_shoot, length=0.05, color='r')

                        ax.set_xlabel('width')
                        ax.set_xlim(-50, 50)
                        ax.set_ylabel('depth')
                        ax.set_ylim(0, 100)
                        ax.set_zlabel('height')
                        ax.set_zlim(-50, 50)

                        #print('max detected speed: ', np.max(np.sqrt(np.array(spd_diff_x)**2 + np.array(spd_diff_y)**2 + np.array(spd_diff_z)**2)))

                        #fig = plt.figure()  # points vector field
                        #ax = fig.gca(projection='3d')
                        ## ax.quiver(position_x, position_y, position_z, speed_x, speed_y, speed_z, length=0.005)
                        #ax.quiver(position_x, position_z, position_y, np.zeros_like(sensitivity), np.zeros_like(sensitivity), sensitivity, length=0.01, color='r')
                        #ax.set_xlabel('width')
                        #ax.set_xlim(-50, 50)
                        #ax.set_ylabel('depth')
                        #ax.set_ylim(0, 100)
                        #ax.set_zlabel('height')
                        #ax.set_zlim(-50, 50)

                    #fig = plt.figure()
                    #plt.scatter(position_x, position_y, c='r', marker='.')
                    #plt.scatter(position_x0, position_y0, c='b', marker='.')

                    break

            #### save values to the last variables

            #if count > 1:
            #    lidar_img_2last = lidar_img_last
            #lidar_img_last = lidar_img_now

            speed_last = speed_now
            Yaw_last = Yaw_now
            pitch_last = pitch
            roll_last = roll
            time_last = time_now

        except KeyboardInterrupt:
            break

    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()
