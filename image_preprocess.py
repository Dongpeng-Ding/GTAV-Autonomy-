import numpy as np
import cv2
import math
import pyflow
import time
import matplotlib.pyplot as plt

from Lidar_tools import polar_to_cartesian, find_lidar_theta_phi_from_coord_Ma



def test_show_lidar_line(image_now_cut, points, focus):
    for point in points:
        camera_coord_x1 = point[0]
        camera_coord_y1 = point[1]
        camera_coord_z1 = point[2]
        camera_coord_x2 = point[3]
        camera_coord_y2 = point[4]
        camera_coord_z2 = point[5]

        x1 = 1099 / 2 - camera_coord_x1 * focus / camera_coord_z1
        # result_y[y, x] =599 / 2 - camera_coord_y * focus / camera_coord_z
        y1 = 299 / 2 - camera_coord_y1 * focus / camera_coord_z1

        x2 = 1099 / 2 - camera_coord_x2 * focus / camera_coord_z2
        # result_y[y, x] =599 / 2 - camera_coord_y * focus / camera_coord_z
        y2 = 299 / 2 - camera_coord_y2 * focus / camera_coord_z2

        cv2.line(image_now_cut, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 250), 4)

    return image_now_cut


def lidar_result_fusion(image, marker, marker_ground_ref):
    marker_tracked = marker['marker_tracked']
    marker_candidate = marker['marker_candidate']
    marker_candidate2 = marker['marker_candidate2']
    marker_s_tracked = marker['marker_s_tracked']
    marker_s_candidate = marker['marker_s_candidate']
    marker_s_candidate2 = marker['marker_s_candidate2']

    def set_color(image_c, marker, value):
        image_c[marker[:, 1], marker[:, 0]] = value
        image_c[marker[:, 1] + 1, marker[:, 0]] = value
        image_c[marker[:, 1] - 1, marker[:, 0]] = value
        image_c[marker[:, 1], marker[:, 0] + 1] = value
        image_c[marker[:, 1], marker[:, 0] - 1] = value
        return image_c

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    rounded_marker_ground = np.round(marker_ground_ref).astype(dtype=int)
    if rounded_marker_ground.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_ground, 0)
        image[..., 1] = set_color(image[..., 1], rounded_marker_ground, 0)
        image[..., 2] = set_color(image[..., 2], rounded_marker_ground, 0)

    rounded_marker_tracked = np.round(marker_tracked).astype(dtype=int)
    rounded_marker_candidate = np.round(marker_candidate).astype(dtype=int)
    rounded_marker_candidate2 = np.round(marker_candidate2).astype(dtype=int)
    if rounded_marker_tracked.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_tracked, 0)
        image[..., 1] = set_color(image[..., 1], rounded_marker_tracked, 0)
        image[..., 2] = set_color(image[..., 2], rounded_marker_tracked, 255)

    if rounded_marker_candidate.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_candidate, 0)
        image[..., 1] = set_color(image[..., 1], rounded_marker_candidate, 255)
        image[..., 2] = set_color(image[..., 2], rounded_marker_candidate, 255)

    if rounded_marker_candidate2.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_candidate2, 0)
        image[..., 1] = set_color(image[..., 1], rounded_marker_candidate2, 150)
        image[..., 2] = set_color(image[..., 2], rounded_marker_candidate2, 255)

    rounded_marker_s_tracked = np.round(marker_s_tracked).astype(dtype=int)
    rounded_marker_s_candidate = np.round(marker_s_candidate).astype(dtype=int)
    rounded_marker_s_candidate2 = np.round(marker_s_candidate2).astype(dtype=int)
    if rounded_marker_s_tracked.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_s_tracked, 255)
        image[..., 1] = set_color(image[..., 1], rounded_marker_s_tracked, 0)
        image[..., 2] = set_color(image[..., 2], rounded_marker_s_tracked, 0)

    if rounded_marker_s_candidate.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_s_candidate, 255)
        image[..., 1] = set_color(image[..., 1], rounded_marker_s_candidate, 200)
        image[..., 2] = set_color(image[..., 2], rounded_marker_s_candidate, 100)

    if rounded_marker_s_candidate2.size != 0:
        image[..., 0] = set_color(image[..., 0], rounded_marker_s_candidate2, 0)
        image[..., 1] = set_color(image[..., 1], rounded_marker_s_candidate2, 255)
        image[..., 2] = set_color(image[..., 2], rounded_marker_s_candidate2, 0)


    #image_monitor = np.concatenate((image_monitor_b, image_monitor_g, image_monitor_r), axis=2)
    #image_monitor = cv2.add(image_monitor, cv2.cvtColor(image_now_cut, cv2.COLOR_GRAY2RGB))

    return image


def opt_flow(image_1, image_2, Mode='all'):
    image_shape = list(image_1.shape)
    image_shape.append(3)
    hsv_shape = tuple(image_shape)
    hsv_0 = np.zeros(hsv_shape)
    hsv_0[..., 1] = 255

    gamma = 3
    #image_1 = cv2.GaussianBlur(image_1, (5, 5), 0.5)
    #image_2 = cv2.GaussianBlur(image_2, (5, 5), 0.5)
    flow = cv2.calcOpticalFlowFarneback(image_1, image_2, None, 0.5, 5,  # pyr_scale, levels
                                        20, 5, 5, 1.2, 0)
    # winsize, iterations, poly_n, poly_sigma, flags

    magnitude, angle = cv2.cartToPolar(-flow[..., 0], -flow[..., 1], angleInDegrees=True)

    if Mode == 'all':
        hsv = hsv_0
        hsv[..., 0] = angle / 2
        hsv[..., 2] = cv2.normalize(magnitude, None, 0, 1, cv2.NORM_MINMAX)
        hsv[..., 2] = 255 * hsv[..., 2] ** (1 / gamma)
        hsv = np.uint8(hsv)
        image_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)

        return magnitude, angle, image_flow
    elif Mode == 'normal':
        return magnitude, angle

    # if cv2.waitKey(5) & 0xFF == ord('q'):
    #    cv2.destroyAllWindows()
    #    break


def opt_flow2(image1, image2):
    image1 = image1.astype(float) / 255.
    image2 = image2.astype(float) / 255.

    image1 = np.expand_dims(image1, axis=2)
    image2 = np.expand_dims(image2, axis=2)

    # for full res
    #alpha = 0.012
    #ratio = 0.5
    #minWidth = 10
    #nOuterFPIterations = 2
    #nInnerFPIterations = 2
    #nSORIterations = 3

    alpha = 0.015
    ratio = 0.5
    minWidth = 10
    nOuterFPIterations = 2
    nInnerFPIterations = 2
    nSORIterations = 3


    colType = 1  # 0 or default:RGB, 1:GRAY (but pass gray image with shape (h,w,1))

    s = time.time()
    u, v, im2W = pyflow.coarse2fine_flow(
        image1, image2, alpha, ratio, minWidth, nOuterFPIterations, nInnerFPIterations,
        nSORIterations, colType)
    e = time.time()
    print('Time Taken: %.2f seconds for image of size (%d, %d, %d)' % (
        e - s, image1.shape[0], image1.shape[1], image1.shape[2]))
    flow = np.concatenate((-u[..., None], -v[..., None]), axis=2)  # reversed direction, from image 2 flow to image 1

    gamma = 3
    hsv = np.zeros([image1.shape[0], image1.shape[1], 3])
    hsv[:, :, 0] = 255
    hsv[:, :, 1] = 255
    mag, ang = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    hsv[..., 0] = ang * 180 / np.pi / 2
    hsv[..., 2] = cv2.normalize(mag, None, 0, 1, cv2.NORM_MINMAX)
    hsv[..., 2] = 255 * hsv[..., 2] ** (1 / gamma)
    hsv = np.uint8(hsv)
    img_flow = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    return mag, ang / np.pi * 180,  img_flow


class Optical_flow:
    def __init__(self):
        self.initialized = 0
        self.image_now = None
        self.image_last = None

    def input(self, image):
        self.image_now = image

    def imshow(self):
        plt.figure()
        plt.subplot(2, 1, 1)
        plt.imshow(self.image_last)
        plt.subplot(2, 1, 2)
        plt.imshow(self.image_now)

    def run(self):
        if self.initialized == 0:
            if self.image_now is not None:
                self.initialized = 1
                self.image_last = self.image_now
            return None, None, None

        mag, dir, img_flow = opt_flow2(self.image_now, self.image_last)
        self.image_last = self.image_now
        return mag, dir, img_flow

def depth_flow(image_1, image_2, dt):
    flow = (image_2 - image_1) / dt
    return flow


def coord_trans(depth_image, Lidar_info, Camera_info1):  # data fusion (transform LiDAR coord to camera coord, then to image coord)

    upper_lim = Lidar_info['upper_lim']
    focus = Camera_info1['focus']
    h_sample = Camera_info1['h_sample']
    v_sample = Camera_info1['v_sample']
    y_sample = Lidar_info['y_sample']
    x_sample = Lidar_info['x_sample']
    x_sample_deg = Lidar_info['x_sample_deg']
    y_sample_deg = Lidar_info['y_sample_deg']

    row_lidar, col_lidar = (np.indices((y_sample, x_sample))).astype('float32')
    theta, phi = find_lidar_theta_phi_from_coord_Ma(row_lidar, col_lidar, x_sample, x_sample_deg, y_sample_deg, upper_lim)

    camera_coord_x, camera_coord_y, camera_coord_z = polar_to_cartesian(depth_image, theta, phi)

    reference_x =(h_sample-1) / 2 - camera_coord_x * focus / camera_coord_z
    reference_y = (v_sample-1) / 2 - camera_coord_y * focus / camera_coord_z

    return reference_x, reference_y


def edge_detect1(image):  # both side, outer edge
    # image = np.uint8(image)
    kernel = [1, -1]
    height = image.shape[0]
    width = image.shape[1]

    # vertical filter
    v_filtered_image_1 = np.empty_like(image)
    for h in range(height):
        if h + 1 < height:
            for w in range(width):
                v_filtered_image_1[h, w] = image[h, w] * kernel[0] + image[h + 1, w] * kernel[1]
                if v_filtered_image_1[h, w] < 0:
                    v_filtered_image_1[h, w] = 0

        else:
            for w in range(width):
                v_filtered_image_1[h, w] = image[h, w] * kernel[0] + image[h, w] * kernel[1]
                if v_filtered_image_1[h, w] < 0:
                    v_filtered_image_1[h, w] = 0

    v_filtered_image_2 = np.empty_like(image)
    for h in range(height):
        if h > 0:
            for w in range(width):
                v_filtered_image_2[h, w] = image[h, w] * kernel[0] + image[h - 1, w] * kernel[1]
                if v_filtered_image_2[h, w] < 0:
                    v_filtered_image_2[h, w] = 0

        else:
            for w in range(width):
                v_filtered_image_2[h, w] = image[h, w] * kernel[0] + image[h, w] * kernel[1]
                if v_filtered_image_2[h, w] < 0:
                    v_filtered_image_2[h, w] = 0

    v_filtered_image = v_filtered_image_2 + v_filtered_image_1

    # horizontal filter
    h_filtered_image_1 = np.empty_like(image)
    for w in range(width):
        if w + 1 < width:
            for h in range(height):
                h_filtered_image_1[h, w] = image[h, w] * kernel[0] + image[h, w + 1] * kernel[1]
                if h_filtered_image_1[h, w] < 0:
                    h_filtered_image_1[h, w] = 0
        else:
            for h in range(height):
                h_filtered_image_1[h, w] = image[h, w] * kernel[0] + image[h, w] * kernel[1]
                if h_filtered_image_1[h, w] < 0:
                    h_filtered_image_1[h, w] = 0

    h_filtered_image_2 = np.empty_like(image)
    for w in range(width):
        if w > 0:
            for h in range(height):
                h_filtered_image_2[h, w] = image[h, w] * kernel[0] + image[h, w - 1] * kernel[1]
                if h_filtered_image_2[h, w] < 0:
                    h_filtered_image_2[h, w] = 0
        else:
            for h in range(height):
                h_filtered_image_2[h, w] = image[h, w] * kernel[0] + image[h, w] * kernel[1]
                if h_filtered_image_2[h, w] < 0:
                    h_filtered_image_2[h, w] = 0

    h_filtered_image = h_filtered_image_1 + h_filtered_image_2

    return v_filtered_image, h_filtered_image

def edge_detect2(image, threshhold):  # color filled
    kernel = [-1, 1]
    height = image.shape[0]
    width = image.shape[1]

    # vertical filter
    v_filtered_image = np.empty_like(image)
    for h in range(height):
        if h + 1 < height:
            for w in range(width):
                v_filtered_image[h, w] = image[h, w] * kernel[0] + image[h + 1, w] * kernel[1]

        else:
            for w in range(width):
                v_filtered_image[h, w] = image[h, w] * kernel[0] + image[h, w] * kernel[1]

    # horizontal filter
    h_filtered_image = np.empty_like(image)
    for w in range(width):
        if w + 1 < width:
            for h in range(height):
                h_filtered_image[h, w] = image[h, w] * kernel[0] + image[h, w + 1] * kernel[1]
        else:
            for h in range(height):
                h_filtered_image[h, w] = image[h, w] * kernel[0] + image[h, w] * kernel[1]

    h_filtered_image2 = np.empty_like(image)
    #threshhold = 20
    for h in range(height):
        value = 0
        h_filtered_image2[h, 0] = value
        for w in range(width - 1):
            if h_filtered_image[h, w] > threshhold:
                value += h_filtered_image[h, w]
                h_filtered_image2[h, w + 1] = value
            elif h_filtered_image[h, w] < -threshhold:
                value += h_filtered_image[h, w]
                h_filtered_image2[h, w + 1] = value
            else:
                h_filtered_image2[h, w + 1] = value

    return h_filtered_image, h_filtered_image2

def edge_detect3(lidar_img):
    #lidar_img = lidar_img / np.max(lidar_img) * 255
    edge_y = cv2.Sobel(lidar_img, cv2.CV_64F, 0, 2, ksize=1)
    edge_x = cv2.Sobel(lidar_img, cv2.CV_64F, 2, 0, ksize=1)
    #edge = np.append(np.expand_dims(edge_y, axis=2), np.expand_dims(edge_x, axis=2), axis=2)
    #edge = np.max(edge, axis=2)
    return edge_y, edge_x


