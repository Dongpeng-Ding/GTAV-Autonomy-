import autograd.numpy as np


def build_roatation_matrix_3D(pitch, roll, yaw):
    base = np.sqrt(np.tan(-pitch) ** 2 + np.tan(-roll) ** 2 + 1)

    rotation_matrix_p_r = np.array([[1 / base, np.tan(-roll) / base, np.tan(-roll) * np.tan(-pitch) * base],
                                          [-np.tan(-roll) / base, 1 / base, np.tan(-pitch) / base],
                                          [np.tan(-roll) * np.tan(-pitch) / base, -np.tan(-pitch) / base,
                                           1 / base]])

    rotation_matrix_y = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0],
                                         [np.sin(yaw), 0, np.cos(yaw)]])

    return rotation_matrix_p_r, rotation_matrix_y

def coords_ro_move(points, trans_matrix, rotation_matrix_pitch_roll, rotation_matrix_yaw, mode):
    if points.size == 0:
        return points
    if points is None:
        return None

    if mode == 'move_first':
        points_centered = points - trans_matrix
        new_points = np.dot(np.linalg.inv(rotation_matrix_yaw.T), points_centered.T)
        new_points = np.dot(rotation_matrix_pitch_roll.T, new_points)
        new_points = new_points.T
        return new_points

    elif mode == 'turn_first':
        new_points = np.dot(rotation_matrix_pitch_roll.T, points.T)
        new_points = np.dot(np.linalg.inv(rotation_matrix_yaw.T), new_points)
        new_points = new_points.T - trans_matrix
        return new_points

    else:
        assert(0),('enter a correct mode')

# AG np
#def coords_ro_move2(points, trans_x, trans_y, trans_z, pitch, roll, yaw, mode): # roll and move the axis not the points
#    if points.size == 0:
#        return points
#    if points is None:
#        return None
#    # calculate rotation matrix
#    base = np.sqrt(np.tan(-pitch) ** 2 + np.tan(-roll) ** 2 + 1)
#
#    rotation_matrix_piro = np.array([[1 / base, np.tan(-roll) / base, np.tan(-roll) * np.tan(-pitch) * base],
#                                     [-np.tan(-roll) / base, 1 / base, np.tan(-pitch) / base],
#                                     [np.tan(-roll) * np.tan(-pitch) / base, -np.tan(-pitch) / base,
#                                      1 / base]])
#
#    rotation_matrix_yaw = np.array([[np.cos(yaw), 0, -np.sin(yaw)], [0, 1, 0],
#                                    [np.sin(yaw), 0, np.cos(yaw)]])
#    # calculate translation matrix
#    move_matrix = np.array([trans_x, trans_y, trans_z])
#
#    if mode == 'move_first':
#        points_centered = points - move_matrix
#        new_coord = np.dot(np.linalg.inv(rotation_matrix_yaw.T), points_centered.T)
#        new_coord = np.dot(rotation_matrix_piro.T, new_coord)
#        new_coord = new_coord.T
#        return new_coord
#
#    elif mode == 'turn_first':
#        new_coord = np.dot(rotation_matrix_piro.T, points.T)
#        new_coord = np.dot(np.linalg.inv(rotation_matrix_yaw.T), new_coord)
#        new_coord = new_coord.T - move_matrix
#        return new_coord
#
#    else:
#        assert(0)


def build_rotation_matrix_2D(delta_yaw):
    rotation_martix = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)],
                                 [np.sin(delta_yaw), np.cos(delta_yaw)]])
    return rotation_martix


def coords_ro_move_2D(points_col, move_matrix, rotation_martix, mode=None):

    if mode == 'move first':
        points_esti_col = np.dot(points_col - move_matrix, np.linalg.inv(rotation_martix))
        return points_esti_col
    elif mode == 'rotation first':
        points_esti_col = np.dot(points_col, np.linalg.inv(rotation_martix)) - move_matrix
        return points_esti_col
    else:
        assert(0), ('enter a correct mode!')


#def move_roatation_2d(points_col, move_x, move_z, delta_yaw, mode=None):
#    move_matrix = np.array([move_x, move_z])  # vehicle move forward
#    roatation_martix = np.array([[np.cos(delta_yaw), -np.sin(delta_yaw)],
#                                 [np.sin(delta_yaw), np.cos(delta_yaw)]])
#    if mode == 'move first':
#        points_esti_col = np.dot(points_col - move_matrix, np.linalg.inv(roatation_martix))
#        return points_esti_col
#    elif mode == 'rotation first':
#        points_esti_col = np.dot(points_col, np.linalg.inv(roatation_martix)) - move_matrix
#        return points_esti_col
#    else:
#        assert(0), ('enter a correct mode!')