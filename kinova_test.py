import os
import sys
import time
import pdb
import math
import pybullet as p
import pybullet_data
from collections import namedtuple
from attrdict import AttrDict
import numpy as np
from utils import update_contact_points, get_robot_ee_pose, rotateMatrixToEulerAngles, rotateMatrixToEulerAngles2, eulerAnglesToRotationMatrix,rotation_matrix_to_quaternion,create_orthogonal_vectors
from camera import ornshowAxes, showAxes, ornshowAxes
from scipy.spatial.transform import Rotation as R
sys.path.append('../')
from pybullet_planning import load_model, connect, wait_for_duration, get_joint_limits, get_max_velocity, get_max_force
from pybullet_planning import  get_movable_joints, set_joint_positions, plan_joint_motion, control_joint, get_joint_positions

HERE = os.path.dirname(__file__)
ROBOT_URDF = os.path.join(HERE, '..', 'data', 'kinova_j2s7s300', 'urdf','j2s7s300.urdf')
OBJECT_URDF = os.path.join(HERE, '..', 'data', 'object_urdf', 'objects', 'block.urdf')

# connect to engine servers
connect(use_gui=True)
# p.setPhysicsEngineParameter(enableFileCaching=0)
# add search path for loadURDF
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# define world
p.setGravity(0, 0, -10) # NOTE
planeID = p.loadURDF("plane.urdf")
tablaID = p.loadURDF("../data/object_urdf/objects/table.urdf",
                            [0.0, 0.0, 0],#base position
                            p.getQuaternionFromEuler([0, 0, 0]),#base orientation
                            useFixedBase=True)
robotStartPos0 = [0.1, 0, 0.2]
robotStartPos1 = [0.1, 0, 0.4]
robotStartPos2 = [0.05, 0, 0.6]

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
relative_offset = np.array([-0.5652227 , -0.7988421 , -0.20585054])
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

robotID  = load_model(ROBOT_URDF, (robotStartPos2, robotStartOrn3), fixed_base=True)

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

gripper_main_control_joint_name = ["j2s7s300_joint_finger_1",
                    "j2s7s300_joint_finger_2",
                    "j2s7s300_joint_finger_3",
                    ]

mimic_joint_name = ["j2s7s300_joint_finger_tip_1",
                    "j2s7s300_joint_finger_tip_2",
                    "j2s7s300_joint_finger_tip_3",
                    ]

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

# open grippe
for i in range(len(gripper_main_control_joint_name)):
    joint = joints[gripper_main_control_joint_name[i]]
    p.setJointMotorControl2(robotID,
                            joint.id,
                            p.POSITION_CONTROL,
                            targetPosition=1,
                            force=joint.maxForce,
                            maxVelocity=joint.maxVelocity)

for i in range(len(mimic_joint_name)):
    joint = joints[mimic_joint_name[i]]
    p.setJointMotorControl2(robotID,
                            joint.id,
                            p.POSITION_CONTROL,
                            targetPosition=1,
                            force=joint.maxForce,
                            maxVelocity=joint.maxVelocity)

for i in range(500):
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

for i in range(len(mimic_joint_name)):
    joint = joints[mimic_joint_name[i]]
    p.setJointMotorControl2(robotID,
                            joint.id,
                            p.POSITION_CONTROL,
                            targetPosition=0,
                            force=joint.maxForce,
                            maxVelocity=joint.maxVelocity)
for i in range(500):
    p.stepSimulation()

start_pose = np.array([[ 7.39879787e-01,  2.51510203e-01, -6.23955548e-01, -1.23662494e-01],
[-8.30135831e-17,  9.27485168e-01,  3.73859942e-01, 7.66715258e-02],
[ 6.72739089e-01, -2.76611418e-01,  6.86227560e-01, 1.46513909e-01],
[ 0.00000000e+00,  0.00000000e+00,  0.00000000e+00, 1.00000000e+00]])

movable_joints = get_movable_joints(robotID)
min_limits = [get_joint_limits(robotID, joint)[0] for joint in movable_joints]
max_limits = [get_joint_limits(robotID, joint)[1] for joint in movable_joints]
max_velocities = [get_max_velocity(robotID, joint) for joint in movable_joints] # Range of Jacobian
current_conf = get_joint_positions(robotID, movable_joints)

robotStartOrn = p.getQuaternionFromEuler([0, 0, np.pi/2])
robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
custom_matrix = np.eye(4)
custom_matrix[:3, :3] = robotStartrot3x3
custom_matrix[:3, 3] = robotStartPos2
jointPose = p.calculateInverseKinematics(robotID, 5, robotStartPos0, robotStartOrn, lowerLimits=min_limits,
                                         upperLimits=max_limits, jointRanges=max_velocities, restPoses=current_conf)

def control_joints_to_target(robotID, jointPose, numJoints, n_steps):
    for i in range(numJoints):
        forcemaxforce = 500 if get_max_force(robotID, i) == 0 else get_max_force(robotID, i)
        p.setJointMotorControl2(bodyUniqueId=robotID,
                                jointIndex=i,
                                controlMode=p.POSITION_CONTROL,
                                targetPosition=jointPose[i],
                                targetVelocity=0.0,
                                force=forcemaxforce,
                                positionGain=0.03,
                                velocityGain=1)
    for i in range(n_steps):
        p.stepSimulation()
        points = update_contact_points()
        check_contact_is_valid(points)
        print("num point:", len(points))
        print("point:", points)

def check_contact_is_valid(points):
        # self.contacts = self.scene.get_contacts()
        contacts = points

        # print("all contact ", self.contacts)
        finger1_contact = False
        finger2_contact = False
        first_timestep_check_contact = 0
        step_length = 0

        for c in contacts:
            bodyidA = c.bodyUniqueIdA
            bodyidB = c.bodyUniqueIdB
            linkIndex1 = c.linkIndexA
            linkIndex2 = c.linkIndexB
            gripper_actor_ids = [6,7,8,9,10,11]
            has_impulse = False

            if (step_length == 1000):
                print("bodyidA, linkIndex1, bodyidA, linkIndex1", bodyidA, linkIndex1, bodyidA, linkIndex1)
            if abs(c.normalForce) > 1e-4:
                has_impulse = True
            if has_impulse and first_timestep_check_contact: # self.gripperJointsInfo
                print("first contact")
                if (bodyidA == 1 and linkIndex1 == -1 and bodyidA == 3 and linkIndex2 in gripper_actor_ids):
                    print("contact ground")
                    return False
            elif has_impulse and not first_timestep_check_contact:
                print("last contact object", step_length)
                if (bodyidA == 1 and linkIndex1 == -1 and bodyidB == 3 and linkIndex2 in gripper_actor_ids):
                    print("contact ground")
                    return False
                elif (bodyidA == 2 and linkIndex1 == -1 and bodyidB == 3 and linkIndex2 in gripper_actor_ids):
                    print("gripper contact object: ", finger1_contact)
                    break
                    # self.finger1_contact = True
                elif (bodyidA == 1 and linkIndex1 == -1 and bodyidB == 2 and linkIndex1 == -1):
                    print("object on ground: ", finger2_contact)
                    break
                    # self.finger2_contact = True
                elif (bodyidA == 1 and linkIndex1 == -1 and bodyidB == 2 and linkIndex1 == -1 and step_length == 1000):
                    print("object on ground at laststep")
                    return False
    
            elif not has_impulse and first_timestep_check_contact:
                print("first not contact object", step_length)

        return True

control_joints_to_target(robotID, jointPose, numJoints-6, 500)

print("end")
# orien = R.from_matrix(start_pose[:3, :3]).as_quat()
# pose = start_pose[:3, 3]
# jointPose = p.calculateInverseKinematics(robotID, 5, pose, orien)