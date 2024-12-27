import os
import sys
import numpy as np
import pyrealsense2 as rs
import cv2
external_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) 
sys.path.append(external_dir) 
from camera import get_rs_pc, point_cloud_flter, ground_points_seg, camera_setup, CameraIntrinsic
import pyvista as pv
import open3d as o3d

class Realsense():

    def __init__(self, width=640, height=480, fps=15):
        self.im_height = height
        self.im_width = width
        self.fps = fps
        self.intrinsics = None
        self.scale = None
        self.pipeline = None
        # bagfile = 'realsense/record/20200901.bag'

        self.connect()
        print("camera init")

    # 连接相机
    def connect(self):

        # configure depth and color streams
        self.pipeline = rs.pipeline()
        config = rs.config()
        # rs.config.enable_device_from_file(config, bagfile, repeat_playback=False)
        config.enable_stream(rs.stream.depth, self.im_width, self.im_height, rs.format.z16, self.fps)
        config.enable_stream(rs.stream.color, self.im_width, self.im_height, rs.format.bgr8, self.fps)
        
        # start streaming
        cfg = self.pipeline.start(config)
        self.scale = cfg.get_device().first_depth_sensor().get_depth_scale()
        print(f"device: {cfg.get_device()}")
        print(f"depth_sensor: {cfg.get_device().first_depth_sensor()}")
        print(f"depth_scale: {self.scale}") # 深度值表示尺度
        print(f"streams: {cfg.get_streams()}") # 显示已启用的所有流信息

        # determine intrinsics
        rgb_profile = cfg.get_stream(rs.stream.color)
        self.intrinsics = self.get_intrinsics(rgb_profile)

        print("---------D435 CONNECT!----------")

    # 获取深度图和彩色图
    def get_data(self):
        # Wait for a coherent pair of frames: depth and color
        frames = self.pipeline.wait_for_frames()

        # align
        align = rs.align(align_to=rs.stream.color)
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        # no align
        # depth_frame = frames.get_depth_frame()
        # color_frame = frames.get_color_frame()

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data(),dtype=np.float32)
        # depth_image *= self.scale # 深度数据转换为实际的物理单位
        depth_image = np.expand_dims(depth_image, axis=2)
        color_image = np.asanyarray(color_frame.get_data())
        return color_image, depth_image
    
    # 绘制深度图和彩色图
    def plot_image(self):
        color_image,depth_image = self.get_data()
        # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

        depth_colormap_dim = depth_colormap.shape
        color_colormap_dim = color_image.shape

        # If depth and color resolutions are different, resize color image to match depth image for display
        if depth_colormap_dim != color_colormap_dim:
            resized_color_image = cv2.resize(color_image, dsize=(depth_colormap_dim[1], depth_colormap_dim[0]),
                                             interpolation=cv2.INTER_AREA)
            images = np.hstack((resized_color_image, depth_colormap))
        else:
            images = np.hstack((color_image, depth_colormap))
        # Show images
        cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
        cv2.imshow('RealSense', images)
        # cv2.imwrite('color_image.png', color_image)
        cv2.waitKey(5000)

    # 获取内参
    def get_intrinsics(self, rgb_profile):
        raw_intrinsics = rgb_profile.as_video_stream_profile().get_intrinsics()
        print("camera intrinsics:", raw_intrinsics)
        # camera intrinsics form is as follows.
        #[[fx,0,ppx],
        # [0,fy,ppy],
        # [0,0,1]]
        # intrinsics = np.array([615.284,0,309.623,0,614.557,247.967,0,0,1]).reshape(3,3) #640 480
        intrinsics = np.array([raw_intrinsics.fx, 0, raw_intrinsics.ppx, 0, raw_intrinsics.fy, raw_intrinsics.ppy, 0, 0, 1]).reshape(3, 3)

        return intrinsics
    
    # 将深度值转换为点云
    def create_point_cloud(self, depth_image, color_image, intrinsics, depth_scale):
        points = []
        colors = []
        
        for v in range(depth_image.shape[0]):
            for u in range(depth_image.shape[1]):
                z = depth_image[v, u] * depth_scale  # 将深度值换算成实际的距离
                if z > 0:  # 仅处理有效的点
                    x = (u - intrinsics.ppx) * z / intrinsics.fx
                    y = (v - intrinsics.ppy) * z / intrinsics.fy
                    points.append([x, y, z])
                    colors.append(color_image[v, u] / 255.0)  # 归一化颜色值

        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(np.array(points))
        point_cloud.colors = o3d.utility.Vector3dVector(np.array(colors))
        
        return point_cloud


if __name__ == "__main__":
    mycamera = Realsense()
    rgb, depth = mycamera.get_data()


    mycamera.plot_image()

    # camera_config = "./setup.json"
    # _, _, _, config = camera_setup(camera_config)

    # camera_intrinsic = CameraIntrinsic.from_dict(config["intrinsic"])  # 相机内参数据

    # pc = get_rs_pc(rgb, depth, camera_intrinsic)
    # cam_XYZA_id1, cam_XYZA_id2, cam_XYZA_pts = point_cloud_flter(pc, depth)
    # # ''' show
    # pv.plot(
    #     cam_XYZA_pts,
    #     scalars=cam_XYZA_pts[:, 2],
    #     render_points_as_spheres=True,
    #     point_size=5,
    #     show_scalar_bar=False,
    # )
    # # '''
    # cam_XYZA_filter_pts, inliers = ground_points_seg(cam_XYZA_pts)
    # # ''' show
    # pv.plot(
    #     cam_XYZA_filter_pts,
    #     scalars=cam_XYZA_filter_pts[:, 2],
    #     render_points_as_spheres=True,
    #     point_size=5,
    #     show_scalar_bar=False,
    # )
    # '''
    # print(mycamera.intrinsics)

 
