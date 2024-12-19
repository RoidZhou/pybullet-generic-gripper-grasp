import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils
import pybullet as p
import cv2
from open3D_visualizer import Open3D_visualizer
from scipy.spatial.transform import Rotation as R
from sapien.core import Pose
from env_kinova import Env
from camera import Camera
# from robots.panda_robot import Robot
from camera import ornshowAxes, Camera, CameraIntrinsic, update_camera_image_to_base, point_cloud_flter, ground_points_seg, rebuild_pointcloud_format
from utils import length_to_plane, get_robot_ee_pose, save_h5, create_orthogonal_vectors, ContactError
import json
from pointnet2_ops.pointnet2_utils import furthest_point_sample
import pyvista as pv
import pcl
import pcl.pcl_visualization
import matplotlib.pyplot as plt
sys.path.append('../')
from pybullet_planning import get_joint_limits, get_max_velocity, get_max_force, get_link_pose
from pybullet_planning import get_movable_joints, set_joint_positions, get_joint_positions, disconnect
cmap = plt.cm.get_cmap("jet")
robotStartPos2 = [0.1, 0, 0.6]

def plot_figure(up, forward, position_world):
    # cam to world
    # up = mat33 @ up
    # forward = mat33 @ forward

    # 初始化 gripper坐标系，默认gripper正方向朝向-z轴
    robotStartOrn = p.getQuaternionFromEuler([0, 0, 0])
    # gripper坐标系绕y轴旋转-pi/2, 使其正方向朝向+x轴
    robotStartOrn1 = p.getQuaternionFromEuler([0, -np.pi/2, 0])
    robotStartrot3x3 = R.from_quat(robotStartOrn).as_matrix()
    robotStart2rot3x3 = R.from_quat(robotStartOrn1).as_matrix()
    # gripper坐标变换
    basegrippermatZTX = robotStartrot3x3@robotStart2rot3x3

    # 计算朝向坐标
    forward = np.array(forward, dtype=np.float32)
    up = np.array(up, dtype=np.float32)
    left = np.cross(forward, up)
    left /= np.linalg.norm(left)
    
    up = np.cross(left, forward)
    up /= np.linalg.norm(up)
    fg = np.vstack([forward, up, left]).T

    # gripper坐标变换
    basegrippermatT = fg@basegrippermatZTX
    robotStartOrn3 = R.from_matrix(basegrippermatT).as_quat()
    ornshowAxes(robotStartPos2, robotStartOrn3)

    rotmat = np.eye(4).astype(np.float32) # 旋转矩阵
    rotmat[:3, :3] = basegrippermatT
    start_rotmat = np.array(rotmat, dtype=np.float32)
    # start_rotmat[:3, 3] = position_world - action_direction_world * 0.2 # 以齐次坐标形式添加 平移向量  ur5 grasp
    start_rotmat[:3, 3] = position_world - forward * 0.17 # 以齐次坐标形式添加 平移向量
    start_pose = Pose().from_transformation_matrix(start_rotmat) # 变换矩阵转位置和旋转（四元数）
    robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)

    rgb_final_pose, depth, _, _ = update_camera_image_to_base(relative_offset_pose, cam)

    rgb_final_pose = cv2.circle(rgb_final_pose, (y, x), radius=2, color=(255, 0, 3), thickness=5)
    Image.fromarray((rgb_final_pose).astype(np.uint8)).save(os.path.join(result_dir, 'viz_target_pose.png'))

# test parameters
parser = ArgumentParser()
parser.add_argument('--exp_name', type=str, help='name of the training run')
parser.add_argument('--model_epoch', type=int, help='epoch')
parser.add_argument('--model_version', type=str, help='model version')
parser.add_argument('--result_suffix', type=str, default='nothing')
parser.add_argument('--device', type=str, default='cuda:0', help='cpu or cuda:x for using cuda on GPU number x')
parser.add_argument('--overwrite', action='store_true', default=False, help='overwrite if result_dir exists [default: False]')
eval_conf = parser.parse_args()

HERE = os.path.dirname(__file__)
ROBOT_URDF = os.path.join(HERE, 'data', 'kinova_j2s7s300', 'urdf', 'j2s7s300.urdf')
OBJECT_URDF = os.path.join(HERE, 'datasets', 'grasp', 'yellow_cup', 'model.urdf')

# load train config
train_conf = torch.load(os.path.join('logs', eval_conf.exp_name, 'conf.pth'))

# load model
model_def = utils.get_model_module(eval_conf.model_version)

# set up device
device = torch.device(eval_conf.device)
print(f'Using device: {device}')

# check if eval results already exist. If so, delete it.
result_dir = os.path.join('logs', eval_conf.exp_name, f'visu_critic_heatmap-model_epoch_{eval_conf.model_epoch}-{eval_conf.result_suffix}')
if os.path.exists(result_dir):
    if not eval_conf.overwrite:
        response = input('Eval results directory "%s" already exists, overwrite? (y/n) ' % result_dir)
        if response != 'y':
            sys.exit()
    shutil.rmtree(result_dir)
os.mkdir(result_dir)
print(f'\nTesting under directory: {result_dir}\n')

# create models
network = model_def.Network(train_conf.feat_dim)

# load pretrained model
print('Loading ckpt from ', os.path.join('logs', eval_conf.exp_name, 'ckpts'), eval_conf.model_epoch)
data_to_restore = torch.load(os.path.join('logs', eval_conf.exp_name, 'ckpts', '%d-network.pth' % eval_conf.model_epoch))
network.load_state_dict(data_to_restore, strict=False)
print('DONE\n')

# send to device
network.to(device)

# set models to evaluation mode
network.eval()


camera_config = "setup.json"
with open(camera_config, "r") as j:
    config = json.load(j)

camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据
# setup camera
# cam = Camera(camera_intrinsic, dist=0.5, fixed_position=False)

# setup env
env = Env()

objectID = env.load_object(OBJECT_URDF)
wait_timesteps = 0
still_timesteps = 0
while still_timesteps < 1000:
    env.step()
    still_timesteps += 1

dist = 0.5
theta = np.random.random() * np.pi*2
phi = (np.random.random()+1) * np.pi/4
pose = np.array([dist*np.cos(phi)*np.cos(theta), \
        dist*np.cos(phi)*np.sin(theta), \
        dist*np.sin(phi)])
relative_offset_pose = pose

# setup camera
cam = Camera(camera_intrinsic, dist=0.5, phi=phi, theta=theta, fixed_position=False)
rgb, depth, pc, cwT = update_camera_image_to_base(relative_offset_pose, cam)
cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth)

# 根据深度图（depth）和相机的内参矩阵来计算相机坐标系中的三维点
# ''' show
pv.plot(
    cam_XYZA_pts,
    scalars=cam_XYZA_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
# '''
cam_XYZA_filter_pts, inliers = ground_points_seg(cam_XYZA_pts)
# ''' show
pv.plot(
    cam_XYZA_filter_pts,
    scalars=cam_XYZA_filter_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
# '''
cam_XYZA_filter_id1, cam_XYZA_filter_id2 = rebuild_pointcloud_format(inliers, cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts)
# 将计算出的三维点信息组织成一个矩阵格式。
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth.shape[0], depth.shape[1])
gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id

# x, y = 270, 270
idx_ = np.random.randint(cam_XYZA_filter_pts.shape[0])
x, y = cam_XYZA_filter_id1[idx_], cam_XYZA_filter_id2[idx_]
# get pixel 3D position (cam/world)
position_world_xyz1 = cam_XYZA[x, y, :3]
position_world = position_world_xyz1[:3]

pt = cam_XYZA[x, y, :3]
ptid = np.array([x, y], dtype=np.int32)
mask = (cam_XYZA[:, :, 3] > 0.5)
mask[x, y] = False
pc = cam_XYZA[mask, :3]
grid_x, grid_y = np.meshgrid(np.arange(448), np.arange(448))
grid_xy = np.stack([grid_y, grid_x]).astype(np.int32)    # 2 x 448 x 448
pcids = grid_xy[:, mask].T
pc_movable = (gt_movable_link_mask > 0)[mask]
idx = np.arange(pc.shape[0])
np.random.shuffle(idx)
while len(idx) < 30000:
    idx = np.concatenate([idx, idx])
idx = idx[:30000-1]
pc = pc[idx, :]
pc_movable = pc_movable[idx]
pcids = pcids[idx, :]
pc = np.vstack([pt, pc])
pcids = np.vstack([ptid, pcids])
pc_movable = np.append(True, pc_movable)
pc[:, 0] -= 5
pc = torch.from_numpy(pc).unsqueeze(0).to(device)

input_pcid = furthest_point_sample(pc, train_conf.num_point_per_shape).long().reshape(-1)
pc = pc[:, input_pcid, :3]  # 1 x N x 3 = [1, 10000, 3]
pc_movable = pc_movable[input_pcid.cpu().numpy()]     # N
pcids = pcids[input_pcid.cpu().numpy()]
pccolors = rgb[pcids[:, 0], pcids[:, 1]]/255
Image.fromarray((rgb).astype(np.uint8)).save(os.path.join(result_dir, 'rgb.png'))

# push through unet
feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F = 10000 x 128
# robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)

# sample a random direction to query
gripper_direction_camera = torch.randn(1, 3).to(device)
gripper_direction_camera = F.normalize(gripper_direction_camera, dim=1)
gripper_forward_direction_camera = torch.randn(1, 3).to(device)
gripper_forward_direction_camera = F.normalize(gripper_forward_direction_camera, dim=1)

up = gripper_direction_camera
forward = gripper_forward_direction_camera
left = torch.cross(forward, up)
left = F.normalize(left, dim=1)

up = torch.cross(left, forward)
up = F.normalize(up, dim=1)

h = length_to_plane(position_world, gripper_forward_direction_camera[0,:].cpu(), plane_height=0.05)
if h > 0.05:
    d_gsp = 0.17
else:
    d_gsp = 0.19 - h
# final_dist = 0.13 # ur5 grasp
final_dist = d_gsp
depth = torch.full((train_conf.num_point_per_shape, 1),final_dist).float().to(device)
plot_figure(up[0].cpu().numpy(), forward[0].cpu().numpy(), position_world)

dirs1 = forward.repeat(train_conf.num_point_per_shape, 1)
dirs2 = up.repeat(train_conf.num_point_per_shape, 1)

# infer for all pixels
with torch.no_grad():
    input_queries = torch.cat([dirs1, dirs2, depth], dim=1)
    net = network.critic(feats, input_queries)
    result = torch.sigmoid(net).cpu().numpy()
    result *= pc_movable

    fn = os.path.join(result_dir, 'pred')
    resultcolors = cmap(result)[:, :3]
    pccolors = pccolors * (1 - np.expand_dims(result, axis=-1)) + resultcolors * np.expand_dims(result, axis=-1)
    o3dvis = Open3D_visualizer(pc[0].cpu().numpy())
    o3dvis.add_colors_map(pccolors)
    # utils.export_pts_color_pts(fn,  pc[0].cpu().numpy(), pccolors)
    # utils.export_pts_color_obj(fn,  pc[0].cpu().numpy(), pccolors)
    # utils.render_pts_label_png(fn,  pc[0].cpu().numpy(), result)

# close env
disconnect()
