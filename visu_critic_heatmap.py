import os
import sys
import shutil
from argparse import ArgumentParser
from PIL import Image
import numpy as np
import torch
import torch.nn.functional as F
import utils
from utils import get_global_position_from_camera
import cv2
from sapien.core import Pose
from env_custom import Env
from camera import Camera
# from robots.panda_robot import Robot
from camera import ornshowAxes, Camera, CameraIntrinsic, update_camera_image_to_base, point_cloud_flter
from utils import control_joints_to_target, get_robot_ee_pose, save_h5, create_orthogonal_vectors, ContactError
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

def plot_figure(up, forward):
    # cam to world
    up = mat33 @ up
    forward = mat33 @ forward

    up = np.array(up, dtype=np.float32)
    forward = np.array(forward, dtype=np.float32)
    left = np.cross(up, forward)
    left /= np.linalg.norm(left)
    forward = np.cross(left, up)
    forward /= np.linalg.norm(forward)
    rotmat = np.eye(4).astype(np.float32)
    rotmat[:3, 0] = forward
    rotmat[:3, 1] = left
    rotmat[:3, 2] = up
    rotmat[:3, 3] = position_world - up * 0.1
    pose = Pose().from_transformation_matrix(rotmat)
    robot.robot.set_root_pose(pose)
    env.render()
    rgb_final_pose, _ = cam.get_observation()
    fimg = Image.fromarray((rgb_final_pose*255).astype(np.uint8))
    fimg.save(os.path.join(result_dir, 'action.png'))

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
OBJECT_URDF = os.path.join(HERE, 'datasets', 'grasp', 'potato_chip_1', 'model.urdf')

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
cam = Camera(camera_intrinsic, dist=0.5, fixed_position=False)

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
positive_mask = cam_XYZA_pts > 0  # 创建布尔掩码
positive_numbers = cam_XYZA_pts[positive_mask] # 选择正数元素

cloud = pcl.PointCloud(cam_XYZA_pts.astype(np.float32))
# 创建SAC-IA分割对象
seg = cloud.make_segmenter()
seg.set_optimize_coefficients(True)
seg.set_model_type(pcl.SACMODEL_PLANE)
seg.set_method_type(pcl.SAC_RANSAC)
seg.set_distance_threshold(0.02)
# 执行分割
inliers, coefficients = seg.segment()
# 获取地面点云和非地面点云
ground_points = cloud.extract(inliers, negative=False)
non_ground_points = cloud.extract(inliers, negative=True)
# 转换为array
cam_XYZA_filter_pts = non_ground_points.to_array()
# ''' show
pv.plot(
    cam_XYZA_filter_pts,
    scalars=cam_XYZA_filter_pts[:, 2],
    render_points_as_spheres=True,
    point_size=5,
    show_scalar_bar=False,
)
# '''
positive_mask = cam_XYZA_filter_pts > 0  # 创建布尔掩码
positive_numbers = cam_XYZA_filter_pts[positive_mask] # 选择正数元素

cam_XYZA_pts_tmp = np.array(cam_XYZA_pts).astype(np.float32)
cam_XYZA_filter_pts_tem = np.array(cam_XYZA_filter_pts).astype(np.float32)

index_inliers_set = set(inliers)
cam_XYZA_filter_idx = []
cam_XYZA_pts_idx = np.arange(cam_XYZA_pts.shape[0])
for idx in range(len(cam_XYZA_pts_idx)):
    if idx not in index_inliers_set:
        cam_XYZA_filter_idx.append(cam_XYZA_pts_idx[idx])
cam_XYZA_filter_idx = np.array(cam_XYZA_filter_idx)
cam_XYZA_filter_idx = cam_XYZA_filter_idx.astype(int)
cam_XYZA_filter_id1 = cam_XYZA_id1[cam_XYZA_filter_idx]
cam_XYZA_filter_id2 = cam_XYZA_id2[cam_XYZA_filter_idx]

# 将计算出的三维点信息组织成一个矩阵格式。
cam_XYZA = cam.compute_XYZA_matrix(cam_XYZA_filter_id1, cam_XYZA_filter_id2, cam_XYZA_filter_pts, depth.shape[0], depth.shape[1])
gt_movable_link_mask = cam.get_grasp_regien_mask(cam_XYZA_filter_id1, cam_XYZA_filter_id2, depth.shape[0], depth.shape[1]) # gt_movable_link_mask 表示为：像素图中可抓取link对应其link_id

x, y = 270, 270
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
pc = pc[:, input_pcid, :3]  # 1 x N x 3
pc_movable = pc_movable[input_pcid.cpu().numpy()]     # N
pcids = pcids[input_pcid.cpu().numpy()]
pccolors = rgb[pcids[:, 0], pcids[:, 1]]

# push through unet
feats = network.pointnet2(pc.repeat(1, 1, 2))[0].permute(1, 0)    # N x F
# robotID = env.load_robot(ROBOT_URDF, start_pose.p, robotStartOrn3)

# sample a random direction to query
gripper_direction_camera = torch.randn(1, 3).to(device)
gripper_direction_camera = F.normalize(gripper_direction_camera, dim=1)
gripper_forward_direction_camera = torch.randn(1, 3).to(device)
gripper_forward_direction_camera = F.normalize(gripper_forward_direction_camera, dim=1)

up = gripper_direction_camera
forward = gripper_forward_direction_camera
left = torch.cross(up, forward)
forward = torch.cross(left, up)
forward = F.normalize(forward, dim=1)

# plot_figure(up[0].cpu().numpy(), forward[0].cpu().numpy())

dirs1 = up.repeat(train_conf.num_point_per_shape, 1)
dirs2 = forward.repeat(train_conf.num_point_per_shape, 1)

# infer for all pixels
with torch.no_grad():
    input_queries = torch.cat([dirs1, dirs2], dim=1)
    net = network.critic(feats, input_queries)
    result = torch.sigmoid(net).cpu().numpy()
    result *= pc_movable

    fn = os.path.join(result_dir, 'pred')
    resultcolors = cmap(result)[:, :3]
    pccolors = pccolors * (1 - np.expand_dims(result, axis=-1)) + resultcolors * np.expand_dims(result, axis=-1)
    utils.export_pts_color_pts(fn,  pc[0].cpu().numpy(), pccolors)
    utils.export_pts_color_obj(fn,  pc[0].cpu().numpy(), pccolors)
    utils.render_pts_label_png(fn,  pc[0].cpu().numpy(), result)

# close env
disconnect()