import os
import pybullet as p
import pybullet_data
from scipy.spatial.transform import Rotation as R
import numpy as np
from bullet_tool import DebugAxes,Camera,CameraIntrinsic,_bind_camera_to_end,update_camera_image
from pybullet_planning import load_pybullet, connect, wait_for_duration
from pybullet_planning import  get_movable_joints, set_joint_positions, plan_joint_motion
import json

'''从matlab开始迁移至pybullet，当前主要完成场景3（似乎还有一点尺寸与matlab中差异）的搭建并调用经典运动规划算法完成运动流程，后续会将自己的运动规划算法加入'''




#连接pybullet 开启UI（第三方库的封装函数，后续挑选使用，尽量使用pybullet自带函数）
connect(use_gui=True)

'''
matlab robot tool 与 pybullet 的Box刚体尺寸转化的区别：
matlab，[0.5,0，0]指的是中心沿着x轴正负延伸0.25，一共延伸0.5
pybullt,[0.5,0，0]指的是中心沿着x轴正负延伸0.5
相同点是都是沿着base中心移动

圆柱转化方式和matlab一致，radius = 0.19,height =0.3，就是代表半径为0.19,并沿着中心上下各延生0.3/2
'''
# 创建碰撞形状
collision_shape_table1 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.5/2,0.6/2,0.05/2])
collision_shape_table2 = p.createCollisionShape(p.GEOM_BOX, halfExtents=[1.3/2,0.4/2,0.05/2])
collision_cylindrical_obstacle = p.createCollisionShape(p.GEOM_CYLINDER, radius=0.19,height=0.6)
collision_cylindrical_cabinet= p.createCollisionShape(p.GEOM_CYLINDER, radius=0.02,height=0.4)
collision_box_cabinet= p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.47/2,0.3/2,0.03/2])
collision_box_workspace= p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.26/2,0.22/2,0.02/2])
collision_box_workspace2= p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.26/2,0.01/2,0.3/2])

collision_box_dynamic= p.createCollisionShape(p.GEOM_BOX, halfExtents=[0.1/2,0.1/2,0.2/2])

# 创建可视化形状
visual_shape1 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.5/2,0.6/2,0.05/2], rgbaColor=[	1,0.92157,0.80392, 1])
visual_shape2 = p.createVisualShape(p.GEOM_BOX, halfExtents=[1.3/2,0.4/2,0.05/2], rgbaColor=[1,0.92157,0.80392, 1])
visual_shape3 = p.createVisualShape(p.GEOM_CYLINDER, radius=0.19,length=0.6, rgbaColor=[1, 0.85, 0.721, 1])
visual_shape4 = p.createVisualShape(p.GEOM_CYLINDER, radius=0.02,length=0.4, rgbaColor=[0.55, 0.55, 0.55, 1])
visual_shape5 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.47/2,0.3/2,0.03/2], rgbaColor=[0.55, 0.55, 0.55, 1])
visual_shape6 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.26/2,0.22/2,0.02/2], rgbaColor=[0.39, 0.58, 0.92, 1])
visual_shape7 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.26/2,0.01/2,0.3/2], rgbaColor=[0.39, 0.58, 0.92, 0.5])
visual_shape8 = p.createVisualShape(p.GEOM_BOX, halfExtents=[0.1/2,0.1/2,0.2/2], rgbaColor=[1, 0, 0, 1])

# 创建一个多体对象，将碰撞形状和可视化形状结合
table1_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_shape_table1,
                            baseVisualShapeIndex=visual_shape1,
                            basePosition=[0.52,0.1,0.18])

table2_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_shape_table2,
                            baseVisualShapeIndex=visual_shape2,
                            basePosition=[0.12,0.6,0.18])

table3_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_obstacle,
                            baseVisualShapeIndex=visual_shape3,
                            basePosition=[0.4,0.5,0.5])

table4_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_cabinet,
                            baseVisualShapeIndex=visual_shape4,
                            basePosition=[0.11,0.52,0.4],
                                          )
table5_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_cabinet,
                            baseVisualShapeIndex=visual_shape4,
                            basePosition=[-0.31,0.52,0.4],
                                          )

table6_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_box_cabinet,
                            baseVisualShapeIndex=visual_shape5,
                            basePosition=[-0.1,0.65,0.4],
                                          )
table7_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_box_cabinet,
                            baseVisualShapeIndex=visual_shape5,
                            basePosition=[-0.1,0.65,0.6],
                                          )
table8_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_box_workspace,
                            baseVisualShapeIndex=visual_shape6,
                            basePosition=[0.45,0.005,0.21],
                                          )
table9_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_box_workspace2,
                            baseVisualShapeIndex=visual_shape7,
                            basePosition=[0.45,0.12,0.34],
                                          )
table10_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_cabinet,
                            baseVisualShapeIndex=visual_shape4,
                            basePosition=[-0.31,0.52,0.4],
                                          )
table11_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_cabinet,
                            baseVisualShapeIndex=visual_shape4,
                            basePosition=[0.11,0.78,0.40],
                                          )
table12_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_cabinet,
                            baseVisualShapeIndex=visual_shape4,
                            basePosition=[-0.310,0.78,0.4],


                                          )

# 动态障碍物（手动开启）
dynamic13_id = p.createMultiBody(baseMass=1.0,
                            baseCollisionShapeIndex=collision_cylindrical_cabinet,
                            baseVisualShapeIndex=visual_shape8,
                            basePosition=[0.250,0.25,0.680],
                                          )

# #显示坐标（测量环境手动开启，环境和matlab还是有微小差异）
# goalPosition1 = DebugAxes()
# goalPosition1.update([0,0.19,0.15
#                       ],[0,0,0,1])


# 自带数据库地址
urdf_root_path = pybullet_data.getDataPath()
# 添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
p.setAdditionalSearchPath(pybullet_data.getDataPath())

#导入ur5 urdf
HERE = os.path.dirname(__file__)
UR_ROBOT_URDF = os.path.join(HERE, 'data', 'universal_robot', 'ur_description', 'urdf', 'ur5.urdf')
#（第三方库的封装函数，后续挑选使用，尽量使用pybullet自带函数）
robot  = load_pybullet(UR_ROBOT_URDF, fixed_base=True)


# * each joint of the robot are assigned with an integer in pybullet（第三方库的封装函数，后续挑选使用，尽量使用pybullet自带函数）
ik_joints = get_movable_joints(robot)

#初始化位置
start_point = [4.91243282232277,
-1.66698724951875,
-1.64678524305916+0.15,
-1.42836894971297,
1.61738012798183,
1.11701446651929]

#让机器人运行到指定关节位置（该指令仅用于调试，因为并未通过电机控制指令，而是初始化指令）
#（第三方库的封装函数，后续挑选使用，尽量使用pybullet自带函数）
set_joint_positions(robot, ik_joints, start_point)


camera_config = "./setup.json"

with open(camera_config, "r") as j:
    config = json.load(j)
camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
camera = Camera(camera_intrinsic)

end_state = p.getLinkState(robot, 0)

# end_state[1]=R.from_euler('XYZ', [-0.35, 0, 0]).as_quat()

update_camera_image(end_state, camera)


#目标位置
goal_point = [3.22515817032901,
-1.85077455516097+0.3,
-1.56710674416374-0.2,
-1.32980615848825-0.3,
1.57232984249436,
0.0519701033343192]
# set_joint_positions(robot, ik_joints, goal_point)



#更改
#第三方库唯一有用的API
path = plan_joint_motion(robot, ik_joints, goal_point, obstacles=[0,1,2,3,4,5,6,7,8,9,10,11,12],
                         self_collisions=True,
                         coarse_waypoints=False)#birrt

for q in path:
    set_joint_positions(robot, ik_joints, q)
    wait_for_duration(0.01)

print('------end------------')