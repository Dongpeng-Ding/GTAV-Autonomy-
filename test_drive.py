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
#import Movement_check3c3_Ma2 as MC3c
import Movement_check4 as MC3c
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
    time_factor = 1 # real time to 1s in game (also change latency in MPC)
    scenario = Scenario(time_scale=1/time_factor, weather='EXTRASUNNY', vehicle='blista', time=[12, 0], drivingMode=[-2, 0, 5.0, 1.0, 1.0],
                        route=Route0)
    #scenario = Scenario(time_scale=0.1, weather='EXTRASUNNY', vehicle='blista', time=[12, 0],
     #                   drivingMode=[0, 0, 3.0, 1.0, 1.0])


    lidar_range = 100
    lidar_X_res = 2
    lidar_Y_res = 2
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


    stoptime = time.time() + 30
    data = []
    count = 0
    start = 0

    while time.time() < stoptime:
        try:

            client.recvMessage()
            t = time.time()
            message = client.recvMessage()


            v = message['speed']
            #a = message['acceleration']
            u_t = message['throttle']
            u_b = message['brake']

            yaw_rate = message['yawRate']
            steering_angle = - message['steering'] / 0.9 # normalize and change to camera coord left is +


            if count > 1:
                dt = t - t_last
                a = (v - v_last) / dt
                data.append([v, a, u_t, u_b, yaw_rate, steering_angle])

            count += 1
            v_last = v
            t_last = t

            if time.time() > stoptime or (cv2.waitKey(20) & 0xFF == ord('q')):
                break

            #if v_last < 1 and v > 1:
            #    start += 1
            if v > 1 and start ==0:
                start += 1
            if start == 1 and v < 1:
                break


        except KeyboardInterrupt:
            break



    # We tell DeepGTAV to stop
    client.sendMessage(Stop())
    client.close()

    # fit vehicle parameters
    drag_x = [] # for fit drags
    drag_y = []
    drag_z = []

    motor_x = [] # for fit drive input
    motor_y = []
    motor_z = []

    brake_x = []  # for fit brake input
    brake_y = []
    brake_z = []

    for line in data:
        if line[2] == 0 and line[3] == 0:
            drag_x.append(line[0])  # v
            drag_y.append(line[2])  # u
            drag_z.append(line[1])  # a
        elif line[3] == 0:
            motor_x.append(line[0])
            motor_y.append(line[2])
            motor_z.append(line[1])
        else:
            brake_x.append(line[0])
            brake_y.append(line[3])
            brake_z.append(line[1])

    drag_x = np.array(drag_x)
    drag_y = np.array(drag_y)
    drag_z = np.array(drag_z)
    motor_x = np.array(motor_x)
    motor_y = np.array(motor_y)
    motor_z = np.array(motor_z)
    brake_x = np.array(brake_x)
    brake_y = np.array(brake_y)
    brake_z = np.array(brake_z)

    drag_x = drag_x[7:] # cut off the latency part of the data
    drag_y = drag_y[7:]
    drag_z = drag_z[7:]
    motor_x = motor_x[7:]
    motor_y = motor_y[7:]
    motor_z = motor_z[7:]
    #brake_x = brake_x[7:]
    #brake_y = brake_y[7:]
    #brake_z = brake_z[7:]

    # a = w0 + w2*v + w3*v^2
    #A = np.array([np.ones_like(drag_x), drag_x, np.sqrt(drag_x)]).T
    #B = drag_z
    #coeff, r, rank, s = np.linalg.lstsq(A, B, rcond=None)
    #print(coeff)

    plt.figure()
    fit = 2 # 1,2,3  drage motor brake
    if fit == 1:
        assert(drag_x.size>2)
        #### fit for drag
        coeff = np.polyfit(drag_x, drag_z, 2)
        print('coeff drag: ', coeff)

        x = np.linspace(0, 20, 100)
        z = coeff[2] + coeff[1] * x + coeff[0] *x *x
        # [-1.03134863e-03 -8.81163528e-03 -3.03185649e+00] refer to power of 2 ,1, 0

        plt.subplot(3,1,1)
        plt.plot(drag_x, drag_z)
        plt.plot(x, z)
    elif fit ==2:
        assert (motor_x.size > 1)
        #### fit for motor control
        coeff = [-1.03134863e-03, -8.81163528e-03, -3.03185649e+00]
        drag_acc = motor_x * motor_x * coeff[0] + motor_x * coeff[1] + coeff[2]

        # acc - drag_acc = k*u/v  constant power
        raw_acc = motor_z - drag_acc
        power = motor_x * raw_acc
        coeff = np.polyfit(motor_y, power, 1)
        print('coeff motor: ', coeff)

        x = np.linspace(0, 1, 10)
        z = coeff[0] + coeff[1] * x
        # [-1.03134863e-03 -8.81163528e-03 -3.03185649e+00] refer to power of 2 ,1, 0

        plt.subplot(3, 1, 2)
        plt.plot(motor_x, power)
        #plt.plot(x, z)

        # max == 7.3
    else:
        assert (brake_x.size > 1)
        #### for brake
        coeff = [-1.03134863e-03, -8.81163528e-03, -3.03185649e+00]
        drag_acc = brake_x * brake_x * coeff[0] + brake_x * coeff[1] + coeff[2]
        raw_acc = brake_z + drag_acc
        coeff = np.polyfit(brake_y, raw_acc, 1)
        print('coeff brake: ', coeff)

        x = np.linspace(0, 1, 10)
        z = coeff[0] + coeff[1] * x
        # [-1.03134863e-03 -8.81163528e-03 -3.03185649e+00] refer to power of 2 ,1, 0

        plt.subplot(3, 1, 2)
        plt.plot(brake_x, raw_acc)
        #plt.plot(x, z)

        # max -23 ~ - 21


