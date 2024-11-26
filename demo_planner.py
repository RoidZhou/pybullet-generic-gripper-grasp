import os
import pybullet as p
import pybullet_data
from pybullet_planning import load_pybullet, connect, wait_for_duration
from pybullet_planning import  get_movable_joints, set_joint_positions, plan_joint_motion



HERE = os.path.dirname(__file__)
UR_ROBOT_URDF = os.path.join(HERE, 'data', 'universal_robot', 'ur_description', 'urdf', 'ur5.urdf')

connect(use_gui=True)



# 自带数据库地址
urdf_root_path = pybullet_data.getDataPath()
# 添加pybullet的额外数据地址，使程序可以直接调用到内部的一些模型
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# workspace= load_pybullet(MIT_WORKSPACE_PATH, fixed_base=True)
robot  = load_pybullet(UR_ROBOT_URDF, fixed_base=True)
table_id = p.loadURDF("table/table.urdf", basePosition=[-1.01, 0, 0])

# * each joint of the robot are assigned with an integer in pybullet
ik_joints = get_movable_joints(robot)
start_point = [0.17195291679571792, -1.2151152721500111, -2.042115573329094,
                              -1.4551581329978485, 1.5707963241241731, 0.17195291700664328]
set_joint_positions(robot, ik_joints, start_point)

goal_point = [0.17195291679571792, -1.3151152721500111, -1.242115573329094,
                              -1.4551581329978485, 1.5707963241241731, 0.17195291700664328]
# set_joint_positions(robot, ik_joints, goal_point)



path = plan_joint_motion(robot, ik_joints, goal_point, obstacles=[1],
                         self_collisions=True,
                         coarse_waypoints=False)

for q in path:
    set_joint_positions(robot, ik_joints, q)
    # wait_if_gui('Continue?')
    wait_for_duration(0.01)