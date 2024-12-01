import os
import time
import pdb
import math
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import numpy as np
from utils import get_robot_ee_pose, rotateMatrixToEulerAngles, rotateMatrixToEulerAngles2, eulerAnglesToRotationMatrix,rotation_matrix_to_quaternion,create_orthogonal_vectors
from camera import ornshowAxes, showAxes, ornshowAxes
from scipy.spatial.transform import Rotation as R
import sys
sys.path.append('../')
from pybullet_planning import load_model, connect, wait_for_duration
from pybullet_planning import  get_movable_joints, set_joint_positions, plan_joint_motion

HERE = os.path.dirname(__file__)
UR_ROBOT_URDF = os.path.join(HERE, '..', 'data', 'franka_description', 'panda_gripper.urdf')
OBJECT_URDF = os.path.join(HERE, '..', 'data', 'object_urdf', 'objects', 'block.urdf')
connect(use_gui=True)

# connect to engine servers
# physicsClient = p.connect(serverMode)
# p.setPhysicsEngineParameter(enableFileCaching=0)
# add search path for loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# define world
p.setGravity(0, 0, -10) # NOTE
planeID = p.loadURDF("plane.urdf")

robotStartPos0 = [0.1, 0, 0.2]
robotStartPos1 = [0.1, 0, 0.4]
robotStartPos2 = [0.1, 0, 0.6]

# 初始化 gripper坐标系，默认gripper正方向朝向-z轴
robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
# gripper坐标系绕y轴旋转-pi/2, 使其正方向朝向+x轴
robotStartOrn1 = p.getQuaternionFromEuler([0, -np.pi/2, 0])
robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
robotStart2rot3x3 = R.from_quat(robotStartOrn1).as_matrix()
# gripper坐标变换
basegrippermatZTX = robotStartrot3x3@robotStart2rot3x3
robotStartOrn2 = R.from_matrix(basegrippermatZTX).as_quat()
# ornshowAxes(robotStartPos0, robotStartOrn2)

# 建立gripper朝向向量relative_offset，[0，0，1]为+z轴方向，由于默认gripper正方向朝向-z轴，所以x轴为-relative_offset
relative_offset = np.array([-0.5652227 , 0.7988421 , -0.20585054])
p.addUserDebugLine(robotStartPos2, robotStartPos2 + relative_offset*1, [0, 1, 0])

# 以 relative_offset 为x轴建立正交坐标系
forward, up, left = create_orthogonal_vectors(relative_offset)
fg = np.vstack([forward, up, left]).T
robotStartOrnfg = R.from_matrix(fg).as_quat()
# ornshowAxes(robotStartPos1, robotStartOrnfg)
print("res: ", np.cross(fg[:, 0], relative_offset))

# gripper坐标变换
basegrippermatT = fg@basegrippermatZTX
robotStartOrn3 = R.from_matrix(basegrippermatT).as_quat()
theta_x, theta_y, theta_z = p.getEulerFromQuaternion(robotStartOrn3)
print("pose, orie0: ", robotStartPos1, robotStartOrn3, theta_x, theta_y, theta_z)
# showAxes(robotStartPos1, basegrippermatT[0], basegrippermatT[1], basegrippermatT[2])
ornshowAxes(robotStartPos2, robotStartOrn3)
print("res: ", np.cross(basegrippermatT[:, 2], relative_offset))

robotID  = load_model(UR_ROBOT_URDF, (robotStartPos2, robotStartOrn3), fixed_base=True)
pose, orie = get_robot_ee_pose(robotID, 5)
theta_x, theta_y, theta_z = p.getEulerFromQuaternion(orie)
print("pose, orie1: ", pose, orie, theta_x, theta_y, theta_z)
ornshowAxes(pose, orie)
'''
# 绕x轴旋转90度
R_AB = np.array([[1,0,0],  # x轴
                 [0,0,-1], # y轴
                 [0,1,0]]) # z轴
'''
# load Object
ObjectID  = load_model(OBJECT_URDF, ([0, 0, 0.10], [0, 0, 0, 1]))

jointTypeList = ["REVOLUTE", "PRISMATIC", "SPHERICAL", "PLANAR", "FIXED"]
numJoints = p.getNumJoints(robotID)
jointInfo = namedtuple("jointInfo",
                       ["id","name","type","lowerLimit","upperLimit","maxForce","maxVelocity"])

joints = AttrDict()
dummy_center_indicator_link_index = 0

# get jointInfo and index of dummy_center_indicator_link
for i in range(numJoints):
    info = p.getJointInfo(robotID, i)
    jointID = info[0]
    jointName = info[1].decode("utf-8")
    jointType = jointTypeList[info[2]]
    jointLowerLimit = info[8]
    jointUpperLimit = info[9]
    jointMaxForce = info[10]
    jointMaxVelocity = info[11]
    singleInfo = jointInfo(jointID, jointName, jointType, jointLowerLimit, jointUpperLimit, jointMaxForce, jointMaxVelocity)
    joints[singleInfo.name] = singleInfo
    # register index of dummy center link
    if jointName == "gripper_roll":
        dummy_center_indicator_link_index = i

gripper_main_control_joint_name = ["panda_finger_joint1",
                    "panda_finger_joint2",
                    ]

mimic_multiplier = [1, 1, 1, -1, -1]

# id of gripper control user debug parameter
# angle calculation
# openning_length = 0.010 + 0.1143 * math.sin(0.7180367310119331 - theta)
# theta = 0.715 - math.asin((openning_length - 0.010) / 0.1143)
gripper_opening_length_control = p.addUserDebugParameter("gripper_opening_length",
                                                0,
                                                0.085,
                                                0.085)
control_dt=1./240.
p.setTimeStep=control_dt
cnt = 1
# open gripper
for i in range(len(gripper_main_control_joint_name)):
    joint = joints[gripper_main_control_joint_name[i]]
    p.setJointMotorControl2(robotID,
                            joint.id,
                            p.POSITION_CONTROL,
                            targetPosition=0.04,
                            force=joint.maxForce,
                            maxVelocity=joint.maxVelocity)
for i in range(50):
    p.stepSimulation()

# close gripper
for i in range(len(gripper_main_control_joint_name)):
    joint = joints[gripper_main_control_joint_name[i]]
    p.setJointMotorControl2(robotID,
                            joint.id,
                            p.POSITION_CONTROL,
                            targetPosition=0,
                            force=joint.maxForce,
                            maxVelocity=joint.maxVelocity)

for i in range(50):
    p.stepSimulation()