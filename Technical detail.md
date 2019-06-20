# GTAV Autonomy 

This is a self driving program based on the game GTAV and the work DeepGTAV and it's variant DeeperGTAV. GTAV is a not opensouce, but still a moddable game. Thus it can be turned into a simulation environment for research. In compare to other opensouce simulaters, GTAV has more realistic grafics and variable environments, which is ideal to develop a personal self driving algorithm for research.

Note: the program is unfinished. More algorithms will be added, current algorithms will be changed in the future.

## Goal of this program

I'm a postgraduate with Master degree of machnical engineering from KIT, with a specialisation in vehicle engineering, but also have 2 years of experence and the final thesis in the field of data analysis during my study. I'm interesting in both conventional vehicle related things and self driving related technologies, especially the core algorithm that makes self driving possible.

The goal of this program is aiming at learning and researching in self driving related algorithms and practice programing skills.

## Core modules and functionality

There are 6 modules in current version:

for preception:
1. Class Ground_esti
2. Class Opt_flow
3. Class Movement_checker
4. Class Lane_finder
for planning:
5. Class Lane_controler
for control (also short term planning):
6. Class Vehicle_controler

The program is mainly focused on perceptions for the virtual car in GTAV. Locolizations, plannings, controls and even mappings will not be fully implemented. 

With the current modules listed above, the virtual car can drive autonomously in GTAV under certain conditions.

### Module Class Ground_esti

This module has 2 taskes: 
calculate pitch and roll of the vehicle.
estimate the drivable area based on based on geometry.

(Note that the algorithms in this module is only a temporary solution to the taskes. A visual based methode is planed to be implemented and add in to the current module.)

The pitch and roll are defined as the angles bewteen the z axis of the vehicle to the surface yz and surface xz which are vertical to the ground when the vehicle has zero pitch and roll. These surfaces rotate with the vehicle in yaw, but keep stationary in pitch and roll, i.e, they are permanently vertical to the ground.

For calculating the pitch and roll, we assume that the 2 lowest rows of lidar samples are on the ground. At each place, center left and right, 3 points from the lowest rows are taken and one vector to the ground can be calculated. The roll and pitch can be derived from each vector. The final roll and pitch are the median from the 3.

### Class Opt_flow

This module estimate the optical flow bewteen 2 images at time t and t+1. The Magnitude and direction of the flow are asigned to the image at time t+1. 

The algorithm of the optical flow is originally from the paper "Beyond Pixels: Exploring New Representations and Applications for Motion Analysis". For details please refer to that paper. The liberay Pyflow used in this module is a python warpper for the c++ implementation of the this algorithm, which was developed for the paper "Learning Features by Watching Objects Move". I adopt this liberay in my work because of its good accuracy. Since this is a CPU implementation, it can hardly reach a short enough computing time for real time performance. I considered looking for a GPU based implementation, but this module is not the key module to be developed in my program, so refining this module is scheduled later.

### Class Movement_checker

Movement is a useful feature for object detection. In comare to other detection methods which use features like shape, color, etc. and make predictions base on a sigle image at time t, the movement based method introduces the dimension of time and makes a prediction from multiple images at different times, and independent from the objects' shape color etc. which means the movement based detection is more robust to unknown object and the generalization seems to be less problematic. This method is expecially important to self driving cars, since the most moving objects are other participants in traffic, they must be detected in order to predict their furture trajectory and avoid collisions. By combining the movement into other features, the robustness of a system can be inceased.

This module classifies the lidar sample into 2 groups: samples belong to stationary objects and samples belong to moving objects. 

Known issue of GTAV: 
Some objects are invisible to lidar: some trees, all leafs, all traffic signs.
 

#### Indepandent moveing object detection

For this task the we first evaluate the 3D position of each pixel at time t and t+1 in camera coordinate. Since the camera matrix, relative position and orientation bewteen camera and lidar is known, the lidar samples can be projected in to image coordinates by using equations porposed in this paper "Robust Fusion of LiDAR and Wide-Angle Camera Data for Autonomous Mobile Robots". By doing a bilinear interpolation the depth of all pixels can be obtained approximately, i.e the 3D positions in camera coordinate can be further calculated. Let Pt+1 be the position of a pixel in image coordinate at time t+1. Its past position Pt is calculated from the optical flow. The depth maps in both time steps is measured, so a estimation of the 3D positions of a same pixel at different time is possiable. (Note that these 3D positions are relatve position to the vehicle itself) The relative speed is than calculated and asigned to Pt+1.

The absolute self movement of the vehicle refer to the ground is calculated from the given speed, yaw rate, pitch rate, roll rate from GTAV and the measured pitch and roll from module Ground_esti. With the absolute movement of the vehicle and the  the relatve movement of the pixels (i.e objects) calculated above, the absolute movement of the pixels related to the ground is obtained.

For less computational cost and less interpolation to reduce error, only those pixels' absolute movement is calculated, whose depth is directly projected or say measured from lidar at time t+1. Then, there are 4 error source of the calculated absolute movement: lidar measurement error in angle and depth, angle error of pitch and roll, optical flow error, interpolation error. The final errors of absolute movement resulted from angle errors are propotional to depth. The optical flow error can also be seen as angel error with repect to its magnitude and direction. the depth error from lidar remain the same. The interpolation error can be divided into two types: 1.unknow geometry between two smples, its error can't be estimated. 2.wrong interpolation position due to angle error, which is also results in an additional error in depth. The second interpolation error is propotional to the gradient of the depth map, which will be called sensitivity later.

To encounter these errors, the threshold to sperate stationary and moving samples is modeled as follows:

threshold_m = w0 + w1 * depth(pt) + w2 * depth(pt) * sensitivity (pt)
threshold_s = threshold_m / w3

In the equation, w0...3 are weights, with w0...2 >= 0, w3 >= 1. The terms are for depth error, angle error, sensitivity error. The absolute speed of a sample related to ground must be exceeds threshold_m to be classified as a moving sample, under threshold_s it will be classified as stationary sample. The gap between threshold and threshold_s is aimed to cut off the uncertain classifications. This gap can filter out some of the wrong classifications, but simultaneously it forms a blind region. Any object which true velocity falls into this gap can't be classified into any of the group. The w3 is currently set to 1. 

Beside the main algorithm above, the unvalid results are excluded during the calculation:
pixels whose depth at time t or t+1 is out of lidar's maximal range
pixels whose position is out of the supporting points of the interpolation, i.e out of lidar's FOV
pixels whose supporting points of the interpolation are likely from different object or on edges. Which is done by thresholding the product of the second derivative of the supporting points on the depth map.

#### Noise filter by tracking

To filter out the noise of classifications and aim for a robust classification result, a pixel can be classified into one of the group only when it was continuously classified in the past n times into the same group. Two types of tracking methods are implemented for comparision:  point to point tracking and point to cloud tracking. Note that the tracking is to the past, not the future.

For a point to point tracking for n times, the 3D positions of pixel Pt and its past Pt-1...t-1-n need to be calculated. As mentioned above assume that the depth of Pt is directly measured from lidar and this pixel is to be classified. The other Pt-1...t-1-n are very unlikely to be perfectly aligned with other lidar projections or measurements in the past due to lidar's low resolution in compare with camera's. So these n pixels are never evaluated before and theirs positions need to be new calculated. Interpolations are iteratively calculated for each past P on its past depth map. If the speed of all Pt...Pt-n are under the threshold_s, the Pt is than classified as stationary point. For moving points, 2 conditions need to be full filled: not only all the speed must be above threshold_m, but the magnitude of all the difference bewteen each pair of the speed vector must also be under threshold_m_t.

Stationary points:
Vpt, ... Vpt-n < threshold_s
Moving Points:
Vpt, ... Vpt-n > threshold_m and
delta Vpt,pt-1, ... delta Vpt-n+1,pt-n < threshold_m_t
where the threshold_m_t has the same form as threshold_m.

If not all or none of the past states meet the the conditions, the Pt will be marked as one of the levels 1 to n candidates of the classification. For example: with n = 4

Vpt, Vpt-1, Vpt-2, Vpt-4 > threshold_m, 
but Vpt-3 < threshold_m 
and,
delta Vpt,pt-1, delta Vpt-2,pt-3, delta Vpt-3,pt-4 < threshold_m_t, 
but delta Vpt-1,pt-2 > threshold_m_t,
this Pt will be marked as level 1 candidate, since it successfully tracked for 1 times.

The disadvantage of this method is that the error of optical flow is cumulated while caluculating each past position. The larger the n is, the bigger the offset between the true Pt-1-n and the estimated P*t-1-n can be statistically. This may result in a larger second type interpolation error depending on the local sensitivity. Even if the interpolation error only cumulates for 2 and 4 times, since we only thresholding the past speed (2 interpolations of positions) and difference of past speed (4 interpolations of positions) for tracking, the interpolation error can also large enough to early stop the tracking for n times, because the local sensitivity also changes with the error of the position. For exsample, the true Pt...t-1-n is a point near the edge of a car, the estimated P*t-1-i (0 < i < 1+n) is out of the edge, where the local sensitivity is very large, that the tracking stopd at i and unable to track for n times. This point Pt will than only become a level i candidate, but it should be tracked and classified.

The point to cloud method tracks the Pt to a past point cloud which is near by the past position of this Pt, in order to conclude that this Pt comes from that cloud, i.e this Pt can be tracked. A threshold_s_t and a threshold_m_t* for tracking stationary points and moving points define the range for detecting the correspondent point cloud. The Euclidean distance between Pt-1 and other points in cloud is compared with the threshold. All in range points from the cloud will be considerd as near by. These 2 threshold have the same form as threshold_m for the same reason. threshold_s_t is currently setted to be equal to threshold_m_t*. Let Pt is the current point, whose speed is > threshold_m or < threshold_s. Its past position Pt-1 will be calculated. Base on the position Pt-1, all the past point clouds are searched for in range points. If no in range points are detected or no in range points have the same type (moving or stationary) as the Pt, Pt will be markered as level 1 candidates of its type. If there are same type in the in range points, the stationary Pt will be marked one level higher then the highest level of the same type candidate in those points or successfully classified if highest same type candidate has level n (track for n times, level starts from 1 to n, i.e n-1 levels) or there is a already classified stationary point in range. For moving Pt, the magnitude of speed difference between Vpt and the speed of each in range point is also compare with threshold_m_t_spd (same form as threshold_m). The moving Pt will level up from the highest level of the same type candidate, whose speed difference with Vpt is also lower than threshold_m_t_spd, or successfully  classified from the level n candidate or a already classified moving point, both with speed difference lower than threshold_m_t_spd. Beside this process, the current point cloud at time t is saved with markers for the next t+1: level 1~n moving or sationary candidate, or classified moving or sationary point. In compare to the point to point tracking, only one position estimation from optical flow and one interpolation involves in the calcullation of the 3D movement at each time step, there is no accumulated error in the point could. At each time step the tracking is preformed by using those point coulds with less error. Also, since a small group of in range points are used for tracking, it is more easier to tracking and successfully classified those point near edges of a object (see the exsample in disadvantage of point to point tracking).

The main disadvantage of the point to cloud tracking is the inhertige of wrong information from those in range points. Assume that Pt is a point on a stationary object A, which due to optical flow or interpolation error is detected as a moving point. It is possible that the in range points near Pt-1 contains not only points from object A or also from other objects. Further, both moving or stationary points can be contained at the same time. In this case the Pt will be level up or classified greedily from the wrong information in the in range point.

To over come this issue, the criteria used for leveling up from the in range points has some alternatives: for moving points campare the median speed of all in range point, for both moving and stationary point level up or classify from the most commen level of the points. But since the in range threshold should set as small as possible in order to get precise and as less as possible near by points to track from, the alternatives above which base on statistical improvment, can be not effective. These method will be further tested.

The both tracking methods can filter out classification noises, but as drawbacks they have letancy before classification. The point to point tracking needs the object can be seen for past n times to make a classification, so it will wait n times until the condition is met. In the point to cloud method, when the object first shows up in the image its pixels have to level up n times to be classified finally.

(p2p more effective to filter noise but less smaples left after classification?
p2c less effective to filter noise but more samples left?)

(all estimate + *)##################################


### Class Lane_finder

This module is about line marker detection  and undertanding for multiple lanes. The goal of the algorithm is to detect line markers and use the detection result to form lanes in the total field of view of the camera with no extra assumption about the orientation and position of the line, so that the vehicle can understand multiple lanes on normal highways, road with split or even at crossing. It aims to lower the dependency on  high precision maps for self driving car or increase the redundancy by understanding the local environment better.  Also this is more like what a humman does while driving, understanding the lanes visually to decide where to dirve in short term, map only helps us to navigate better.

The class Lane_finder has following steps:
Image preprocessing
Light condition compensation on the road
Lane marking detection
Lane marking postprocessing
Understanding lanes from markings

Known issue of GTAV: lane line textures that far from the vehicle, are "washed out"

#### Image preprocess

A pixel wise mask for region of interest is first generated from the driveable area estimated from module Ground_esti.

To detect white and yellow lane marking, the original 3 channel image is converted to singal channel image as follows.

For detecting white marking:
I_intensity = min(I_r, I_g, I_b)

For detecting yellow marking:
I_yellow/blue = (I_r + I_g - I_b + 255) / 2

For each of the single channel images, a mean value inside the ROI mask is calculated. This value is then used in the next step "Compensating shadows" to adjust the range of the histogram of the image.

Then, the I_intensity, I_yellow/blue and the mask ROI are warpped to birdeye's view. This makes the lane markings appears to be the same size at different distance. The warp matrix at each time t is calculated from the current lidar samples for more precise warpping results.

Note that although the the algorithm have the ability distinguish yellow and white lines, but those infomation is not furhter used currently. This part will be implemented in the furture to better understand lanes and control the vehicle.

#### Light condition compensation on the road

Compensating the effect of shadows or light variations is a important task for robust line detection algorithm. A very simple but effective algorithm is performed.

Lane marking itself has alway the same width due to regulation, so they can be treated as a "salt" noise (in "salt and peper" noise) on the road and removed by a median filter. The median filter also removes "peper" noise, but this kind of noise is not lane marking, since lane marking often has reletivly high reflection than the background under the same light condition. The "peper" noise can be thin shadows seen in the figure below.

If a pixel in the median filtered image is lighter than its original value, it will be restore it original value. In this way only "salt" noise (potential lane marking) is filtered and removed from the original image, the filtered image is then become a background with different light condition but without lane marking. As mentioned above, a lane marking should stands out from it background under the same light condition. So a subtraction between original image and filtered image is done and results in a difference image. The pixels with high difference is highly probably from a lane marking.

figure #####################################################

The left image is the I_intensity for detcting white lane marking. The right image is the difference image after the process. Note that this methode evaluate the potental lane marking pixels regardless of the light condition. The lane marking inside the shadow is also recoverable. Faulty evaluation appears if any light spots (upper right corner, due to shadows of a tree; center front, due to a white paint on the road) or sharp corner where light suround by shadow (lower left lane at the egde of shadows) is smaller than the kernel size of the median filter.

#### Lane marking detection

Both images I_intensity and I_yellow/blue corrected from the above step are convert to binary images by applying a adaptive threshold. (since light conditions are already compensate, a less computational expensive global threshold may be enough) Then the binary images is masked and line detection by using Hough transform is performed.

The Hough transform here is from opencv, which only detect straigh lines. A curved lane marking will be detected as many short lines. This problem will be encoutered in the following postprocessing step. 

A direct curve detection is also possible by Hough transform, but it is not used due to 3 consideration:
1. higher computational cost due to higher dimension in parameter space.
2. higher computational cost for following algorithms, since more computations are needed to handle nonlinearity of the  curves.
3. the observed final cross-track and anlge errors of the vehicle, which are partly due to the approximation using straight lines, are acceptable.

After the line detection on warped image, all lines are warpped back to the original image. Using the same methode in class movement_checker, the 3D position of the line vertices are estimated with the depth map. The effect from vehicle's pitch and roll on the 3d positions are eliminated by changing the coordinate system related to yaw only. 

These lines are also checked with their consistency at time t and time t-1. All current lines are tranformed back to time t-1 using dt, speed, acceleration, yawrate of the vehcle and campared with previours detected lines. If a line can't be tracked from time t-1, it will be excluded as a noise. 

#### Lane marking postprocessing

In the postprocessing step all detected lines are clustered into different groups of lane markings. This encounter the issue that one lane marking, staight or curved, is detected as multiple lines from Hough transform in opencv.

The cluster algorithm is inspired from DBSCAN cluster method and based on the recursion. The recursion works in a way that the informations flows not only top-down but also bottem-up, i.e the lower level of recursions not only uses informations from upper level to do calculations, but aslo returns infos back to upper level. This both side 

At the top level of the recursion, the algorithm loops for all n detected lines. Inside each loop, a next level of recursion starts with a judgement, if the current line has not checked before. if true, it will be checked for connetion to other lines(belong to same lane marking). This results new conneted lines to the current line. These new connetions are then add to older connetions from upper level of recursion in order to recorde all connections and will be eventially sent back to upper level. After teh connection check, the algotithm loops for all new conneted lines and starts next level recursions. In each recursion, if the current line is checked before and has assigned with a cluster number already, this number will be inherited and send back to the upper level of the recursion eventially. In this case, no further connecton checks are needed and no next level recursion needs to be performed. The recursion will stop until all lines are checked. 

At the top level of the recursion, the recorded connections and inherited cluster number are recieved. All lines under the recorded connections will be marker with the inherited cluster number if it exist. If there is no inheritage, those lines will be marker with a new cluster number.

All the detected lines are clustered into groups of lane markings. To approximate each lane marking, these lines in the cluster are refitted with one straight line.

To handle Y shape lines at a road Y slpit. The rms error of the fitting result is calculated. If it bigger than a threshold, the lines on the left and right of the current fitting result are speratly refitted with 2 new lines to approximate the Y shape. This process loops until the rms error smaller than the threshold or the maximum loop number is reached. This process inspired by the RANSAC methode.

#### Understanding lanes from markings

In this step, the algorithm forms lanes between lane markings. The width bewteen each pair of lane markings are calculated. The 2 lane markings are often not parallel, so the distance is defined as follows:

Figure#####################################################

The verticies a,b,c,d forms 2 lines refer to the position of 2 lane markings. Point i and j are the centerpoint of the lines. Point k2 is the center point of the line ij. Point k1 is the intercept point of the lines ab and cd. The width between 2 lane markings is defined as the length of the line v1v2, which is vertical to line k1k2 at point k2.

The algortihm checks for each lane marking it nearest left and right neighbors based on the width. Each pair of nearest neighbors can be recognized as a lane, when it full fills the following conditions:

angle bewteen them < threshold_a
threshold_w1 < width < threshold_w2
the neighbors are not off setted to each other (see figure below)

Figure################################################

### Class Lane_controler

This class is a finite state machine to decide which lane should keep in to. Currently it only contains 2 verly simple states:

1. find and drive to nearest lane
2. keep to the same lane
  2.1 keep to inertial lane

In the first state, the distances bewteen each lane and the vehicle is compared to find out the nearest lane. This lane is setted to be the target lane. The distance is calculated in parameterspace of the straight line and defined as follows:

distance = w1 * rho + w2 * phi
where w1 and w2 are weights.

In the second state, the previous target lane at time t-1 is predicted into current time t by using the yawrate, speed, longitudinal acceleration and dt. This inertially predicted  lane in cartesian coordinate is then compare with all current lanes in parameterspace (theta, rho) in order to find the same lane at time t as the new target lane.

If there is no related lane found at time t, the inertially predicted lane itself is considered as the target lane at time t. For safty, the inertial lane can be used only for n times, at the time t+n, the FSM will transform to the first state.

(The target lane is in cartesian coordinate and represent by its 2 vertices)

Beside the process above, a extended kalman filter is used to smooth the target lane in its parameterspace theta, rho. The none-filtered target lane in cartesian coordinate is used for its postions and length information. The filtered target lane in theta-rho space is used for the cross-track distance and angle. 

Note that due to the line detection process, the verticies of the lane at different time steps are often not following any dynamic behaviour, so the verticies can't be filtered with kalman filter. Exceptions is that the full lane is covered in the image and perfectlly detected through the process, but this is often not the case.  


### Class Vehicle_controler

This module is a model predictive controler to dirve the vehicle.

#### Vehicle model: longitudinal behaviour

For the longitudinal driving, the vehicle's engine is modeled as a electic motor with constant maximum power and instant reaction to input. The power output is linearly propotional to driving input.

P = k'*u (0<= u <=1, 0<= P <=Pmax)

The mechanical drag of the vehicle itself (include tire's rolling drag) is approximated by a linear term and a constant. All internal and external air drags are approximated by a quadratically proportional term. The "drag" from the slope of the road is represented by a variable s.

drag = s' + w0' + w1'*v + w2'*v^2

By taking those terms into force blance equation and dividing the mass on both side:
acc = k*u/v + s + w0 + w1*v + w2*v**2 
(acc is measured as dv/dt, acc sensor is not very reliable)

For simplicity, the parameter w0...3 are measured together from rolling tests with u==0, and calculated by solving a least squre problem. Then set u !=0 and measures and calculate the k in the same way. The k is measured and calculated seperatly, because the engine model is much less precise than the drag model, its error should not affect the parameter w0...3. 

The variable s is then the only unknown value, which can be calculated while driving using the measurements of v, u, t.

#### Vehicle model: lateral behaviour
The steering part of the vehicle model is a simple bicycle with no lateral slip of the tires (slip angle == 0). In GTAV the steering input to steering wheel angle is a unknown function. Even if a series of functions: steering wheel angle to front wheel angle, front wheel angle to slip angle, slip angle to lateral force are unknown, it is still unable to model the steering precisely without the first function. For simplicity, the total function of the steering input to yawrate is currently modeled as linear function.

#### Environment model

To avoid collision to other objects, the distance is aslo took into account in the MPC as a part of short term planning. There are still some missing modules currently, i.e the current algorithm doesn't build a map from the lidar samples, nor does a classification to classify the samples into concrete object. On the other hand, each lidar sample is directly used to represent the environment. (This will lead to higher computational cost and the movement of moving objects can't be estimated with precisions.) Those modules will be added in the furture.

The lidar samples are in vehicle coordinate. In each furture state in the MPC, the moving lidar samples are first moved with their speed estimated from module movement checker. Then, the coordinate system is changed to the next vehicle's position.

Beside the lidar samples, the vertices of the selected lane from module Lane_controler is also updated in each state by changing the coordinate.

#### Error function

The error function is:

Error = sum for all states(w0*cross_track_error + w1*angle_error + w*2speed_error + w3*distance_error + w4*L2_penalization_longitudinal_control + w5*L2_penalization_steering_control)

Beside the distance_error, all other error terms are quadraticall and calculated as the difference bewteen the target value and true value. The penalization terms are also quadraticall and calculated as the difference between last value and current value.

The term distance_error as follows:

All lidar samples at a state are checked with the "in range" condition: 

sample_x < max_range and 
abs(sample_y) < max_width and 
sample_z < max_height and 
sample is a road marker (from modul ground_checker)

The in range condition prevent this error term from disturbing the behaviour of MPC controler when the vehicle are not facing collision risk. These in range samples are then used to calculated the distance_error term.

distance_error  = max(exp(w0 - sample_x + w1 * (w2 - abs(sample_y)))) * abs(speed)

## To do 

Finish classes Lane_controler and Vehicle_controler.
Deep lerning based scene understanding for drivable area, traffic signs, road markers etc.
Lidar sample classification.
Build a more general finite state machine to overall. control the vehicle and give it other functionality.
Refine the current algorithm in classes Movement_checker and Lane_finder.
Imporve code style and performence, may switch to c++.
