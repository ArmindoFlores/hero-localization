#!/usr/bin/env python3

import sys

import numpy as np
import open3d
import rospy
import sensor_msgs.point_cloud2 as pc2
import std_msgs.msg
from nav_msgs.msg import Odometry
from sensor_msgs.msg import PointCloud2


def open3d_to_ros(pc_o3d: open3d.geometry.PointCloud, frame_id: str):
    """Convert an Open3D point cloud to a numpy structured array"""
    np_points = np.asarray(pc_o3d.points)
    fields = [
        pc2.PointField(name="x", offset=0, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name="y", offset=4, datatype=pc2.PointField.FLOAT32, count=1),
        pc2.PointField(name="z", offset=8, datatype=pc2.PointField.FLOAT32, count=1),
    ]
    header = std_msgs.msg.Header()
    header.frame_id = frame_id
    header.stamp = rospy.Time.now()
    cloud_msg = pc2.create_cloud(header, fields, np_points)
    return cloud_msg

class AdaptiveVoxelizationNode:
    def __init__(self, node_name="adaptive_voxelization"):
        rospy.init_node(node_name, anonymous=True)

        self.pcd_path = rospy.get_param("~REFERENCE_POINT_CLOUD_PATH", None)
        self.publish_map_rate = rospy.get_param("~PUBLISH_MAP_RATE", 1)
        self.publish_global_rate = rospy.get_param("~PUBLISH_GLOBAL_MAP_RATE", 0.1)
        self.map_topic_name = rospy.get_param("~MAP_TOPIC", None)
        self.global_map_topic_name = rospy.get_param("~GLOBAL_MAP_TOPIC", None)
        self.robot_pose_topic = rospy.get_param("~ROBOT_POSE", None)
        self.map_frame = rospy.get_param("~MAP_FRAME", None)
        self.z_clip_distance_below = rospy.get_param("~Z_CLIP_DISTANCE_BELOW", 1e128)
        self.z_clip_distance_above = rospy.get_param("~Z_CLIP_DISTANCE_ABOVE", 1e128)
        self.cloud_id = None
        self.robot_position = None
        self.should_update_map = True

        if self.pcd_path is None:
            rospy.logerr("Parameter '~REFERENCE_POINT_CLOUD_PATH' is required.")
            raise SystemExit(1)
        
        if self.robot_pose_topic is None:
            rospy.logerr("Parameter '~ROBOT_POSE' is required.")
            raise SystemExit(1)
        
        if self.map_topic_name is None:
            rospy.logerr("Parameter '~MAP_TOPIC' is required.")
            raise SystemExit(1)
        
        if self.map_frame is None:
            rospy.logerr("Parameter '~MAP_FRAME' is required.")
            raise SystemExit(1)
        
        # Load reference point cloud
        rospy.loginfo(f"Loading reference point cloud from {self.pcd_path}")
        self.reference_point_cloud = open3d.io.read_point_cloud(self.pcd_path)
        if self.reference_point_cloud.is_empty():
            rospy.logerr("Loaded point cloud is empty.")
            return 1
        
        self.global_point_cloud = open3d_to_ros(self.reference_point_cloud, self.map_frame)
        self.downsampled_point_cloud = open3d_to_ros(self.reference_point_cloud, self.map_frame)
        
        # Subscribe to the robot's position estimate
        rospy.Subscriber(self.robot_pose_topic, Odometry, self.update_reference, queue_size=1)

        # Create a new publisher for the reference point cloud
        self.map_publisher = rospy.Publisher(self.map_topic_name, PointCloud2, queue_size=1)
        rospy.Timer(rospy.Duration(1.0 / self.publish_map_rate), self.publish_map)

        if self.global_map_topic_name is not None:
            # Create a new publisher for the global reference point cloud
            self.global_map_publisher = rospy.Publisher(self.global_map_topic_name, PointCloud2, queue_size=1)
            rospy.Timer(rospy.Duration(1.0 / self.publish_global_rate), self.publish_global_map)
        else:
            self.global_map_publisher = None

    def update_reference(self, msg: Odometry):
        position = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            0
        ])
        if self.robot_position is None or np.linalg.norm(self.robot_position - position) > 1:
            rospy.loginfo(f"Updated robot position: {position}")
            self.robot_position = position
            self.should_update_map = True

    def adaptive_voxel_downsample(self, min_voxel_size=0.25, max_voxel_size=0.75, radius_thresholds=(15, 30.0)):
        # Extract all points
        points = np.asarray(self.reference_point_cloud.points)

        # Apply vertical filter: only keep points within z_clip_distance meters (in Z-axis) from the robot
        vertical_mask = (points[:, 2] - self.robot_position[2] <= self.z_clip_distance_above) & (self.robot_position[2] - points[:, 2] <= self.z_clip_distance_below)
        z_clipped = self.reference_point_cloud.select_by_index(np.where(vertical_mask)[0])

        # Recompute distances after vertical filtering
        z_points = np.asarray(z_clipped.points)
        distances = np.linalg.norm(z_points - self.robot_position, axis=1)

        # Assign bins based on horizontal distance
        near_mask = distances < radius_thresholds[0]
        mid_mask = (distances >= radius_thresholds[0]) & (distances < radius_thresholds[1])
        far_mask = distances >= radius_thresholds[1]

        # Split into three distance-based subsets
        pcd_near = z_clipped.select_by_index(np.where(near_mask)[0])
        pcd_mid = z_clipped.select_by_index(np.where(mid_mask)[0])

        # Apply voxelization with adaptive resolution
        pcd_near = pcd_near.voxel_down_sample(voxel_size=min_voxel_size)
        pcd_mid = pcd_mid.voxel_down_sample(voxel_size=max_voxel_size)

        # Combine results
        return pcd_near + pcd_mid

    def publish_map(self, _):
        if self.robot_position is not None and self.should_update_map:
            self.should_update_map = False
            self.downsampled_point_cloud = open3d_to_ros(self.adaptive_voxel_downsample(), self.map_frame)
        self.map_publisher.publish(self.downsampled_point_cloud)
    
    def publish_global_map(self, _):
        rospy.loginfo("Publishing global point cloud")
        self.global_map_publisher.publish(self.global_point_cloud)

def main():
    node = AdaptiveVoxelizationNode()

    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass