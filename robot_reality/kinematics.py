import numpy as np


ROBOT_WIDTH, ROBOT_LENGTH = 0.175, 0.4387  # 机器人长宽
L1, L2, L3 = 0.037, 0.25, 0.25  # consistent with the manual
FR, FL, RR, RL = 0, 1, 2, 3


def rotate_matrix(axis, theta):
    if axis == 'x':
        matrix = [
            [1, 0, 0, 0],
            [0, np.cos(theta), -np.sin(theta), 0],
            [0, np.sin(theta), np.cos(theta), 0],
            [0, 0, 0, 1]
        ]
    elif axis == 'y':
        matrix = [
            [np.cos(theta), 0, np.sin(theta), 0],
            [0, 1, 0, 0],
            [-np.sin(theta), 0, np.cos(theta), 0],
            [0, 0, 0, 1]
        ]
    elif axis == 'z':
        matrix = [
            [np.cos(theta), -np.sin(theta), 0, 0],
            [np.sin(theta), np.cos(theta), 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ]
    else:
        assert 0
    return np.array(matrix)


def translation_matrix(dx, dy, dz):
    matrix = [
        [1, 0, 0, dx],
        [0, 1, 0, dy],
        [0, 0, 1, dz],
        [0, 0, 0, 1]
    ]
    return np.array(matrix)


def get_toe_position(motor_angle):
    pos = []
    for i in range(4):
        pos.extend(compute_toe_position(toe_id=i, motor_angle=motor_angle).tolist())
    return list(pos)


def compute_toe_position(toe_id, motor_angle):
    """
    Compute the position of foot
    :param toe_id: the index of legs
    :param motor_angle: current motor angle
    :return: The position in base frame
    """
    motor_angle = motor_angle[toe_id * 3: toe_id * 3 + 3]  # current angle
    matrices = []
    flag = 1  # 1 if left, -1 if right

    if toe_id == FR:
        matrices.append(translation_matrix(ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2, 0))  # 从躯干中心坐标系平移到髋
    elif toe_id == FL:
        matrices.append(translation_matrix(ROBOT_LENGTH / 2, ROBOT_WIDTH / 2, 0))
    elif toe_id == RR:
        matrices.append(translation_matrix(-ROBOT_LENGTH / 2, -ROBOT_WIDTH / 2, 0))
    elif toe_id == RL:
        matrices.append(translation_matrix(-ROBOT_LENGTH / 2, ROBOT_WIDTH / 2, 0))
    else:
        assert 0

    matrices.extend([
        rotate_matrix('x', motor_angle[0]),  # 旋转使y轴与大腿电机垂直
        translation_matrix(0, flag * L1, 0),  # 从髋电机转移到大腿电机
        rotate_matrix('y', np.pi / 2 + motor_angle[1]),  # 旋转到x轴与大腿同向
        translation_matrix(L2, 0, 0),  # 沿大腿平移
        rotate_matrix('y', -np.pi / 6 + motor_angle[2]),  # 在腿关节旋转，直到x轴与小腿同向
        translation_matrix(L3, 0, 0)  # 沿小腿平移
    ])
    matrix = matrices[0]
    for tmp_m in matrices[1:]:
        matrix = np.matmul(matrix, tmp_m)  # 动坐标系，矩阵顺序应该是从左往右
    pos = np.array([[0], [0], [0], [1]])
    ret = np.matmul(matrix, pos)
    assert ret[-1] == 1
    return ret[: -1].reshape(-1)
