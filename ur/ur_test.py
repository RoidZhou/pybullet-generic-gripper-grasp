import os
import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs


def depth_to_pointcloud(depth_image, intrinsic):
    # Create Open3D Image from depth map
    o3d_depth = o3d.geometry.Image(depth_image)

    # Get intrinsic parameters
    fx, fy, cx, cy = intrinsic.fx, intrinsic.fy, intrinsic.ppx, intrinsic.ppy

    # Create Open3D PinholeCameraIntrinsic object
    o3d_intrinsic = o3d.camera.PinholeCameraIntrinsic(width=depth_image.shape[1], height=depth_image.shape[0], fx=fx,
                                                      fy=fy, cx=cx, cy=cy)

    # Create Open3D PointCloud object from depth image and intrinsic parameters
    pcd = o3d.geometry.PointCloud.create_from_depth_image(o3d_depth, o3d_intrinsic)

    return pcd


def save_pointcloud(pcd, file_name):
    o3d.io.write_point_cloud(file_name, pcd)


# 初始化 RealSense 相机
pipeline = rs.pipeline()
config = rs.config()
config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 5)
config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 5)
profile = pipeline.start(config)

align_to = rs.stream.color
align = rs.align(align_to)

cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
depth_profile = rs.video_stream_profile(profile.get_stream(rs.stream.depth))
depth_intrinsics = depth_profile.get_intrinsics()

if not os.path.exists('data'):
    os.makedirs('data')

subfolders = ['images', 'depths', 'point_clouds']
for folder in subfolders:
    folder_path = os.path.join('data', folder)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

counter = 0

try:
    while True:
        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        if not aligned_depth_frame:
            continue

        depth_frame = frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()
        if not aligned_depth_frame:
            continue

        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        color_intrinsics = color_frame.profile.as_video_stream_profile().intrinsics
        depth_intrinsics = aligned_depth_frame.profile.as_video_stream_profile().intrinsics
        pc = depth_to_pointcloud(depth_image, depth_intrinsics)

        cv2.imshow('RealSense', color_image)

        depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.008), cv2.COLORMAP_JET)
        cv2.imshow('depth_color', depth_colormap)

        # 检查是否按下了 's' 键，如果按下了，就保存当前帧的 RGB、深度图和点云
        key = cv2.waitKey(1)
        if key == ord('s'):
            # 保存 RGB 图像
            rgb_file_path = os.path.join('images', 'add_{:04d}.jpg'.format(counter))
            cv2.imwrite(rgb_file_path, color_image)
            print('color saved', rgb_file_path)

            # 保存深度图像
            depth_file_path = os.path.join('depths', 'depth_{:04d}.png'.format(counter))
            cv2.imwrite(depth_file_path, depth_image)
            print('depth saved', depth_file_path)

            # 将点云保存为 pcd 文件
            #pcd_file_path = os.path.join('point_clouds', 'point_cloud_{:04d}.pcd'.format(counter))
            #save_pointcloud(pc, pcd_file_path)
            #print('pc saved', pcd_file_path)

            # 更新计数器
            counter += 1

        # 检查是否按下了 ESC 键，如果按下了，就退出循环
        if key == 27:
            cv2.destroyAllWindows()
            break
finally:
    pipeline.stop()