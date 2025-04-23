#!/usr/bin/env python3

import sys
import typing

import numpy as np
import open3d
import ros_numpy
import rospy
import tf
import tf.transformations
import tf.transformations as tf_trans
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2


def angle_diff(m1, m2):
    q1 = tf.transformations.quaternion_from_matrix(m1)
    q2 = tf.transformations.quaternion_from_matrix(m2)
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)
    
    dot = np.clip(np.abs(np.dot(q1_norm, q2_norm)), 0.0, 1.0)
    return 2 * np.arccos(dot)

def transformation_to_pose(transform, frame_id):
    """Convert a 4x4 transformation matrix (from Open3D ICP) to a ROS Pose message."""

    # Extract translation
    translation = transform[0:3, 3]

    # Convert rotation to quaternion
    quaternion = tf.transformations.quaternion_from_matrix(transform)

    # Create Pose message
    pose = PoseWithCovarianceStamped()
    pose.header.stamp = rospy.Time.now()
    pose.header.frame_id = frame_id

    pose.pose.pose.position.x = translation[0]
    pose.pose.pose.position.y = translation[1]
    pose.pose.pose.position.z = translation[2]

    pose.pose.pose.orientation.x = quaternion[0]
    pose.pose.pose.orientation.y = quaternion[1]
    pose.pose.pose.orientation.z = quaternion[2]
    pose.pose.pose.orientation.w = quaternion[3]
    # Temporary: set covariances
    pose.pose.covariance = [
        1e-4, 0, 0, 0, 0, 0,
        0, 1e-4, 0, 0, 0, 0,
        0, 0, 1e6, 0, 0, 0,
        0, 0, 0, 1e6, 0, 0,
        0, 0, 0, 0, 1e6, 0,
        0, 0, 0, 0, 0, 1e-3
    ]

    return pose

def position_and_quaternion_to_matrix(px, py, pz, qx, qy, qz, qw):
    # Create 4x4 transformation matrix from quaternion and translation
    matrix = tf_trans.quaternion_matrix([qx, qy, qz, qw])  # returns 4x4 rotation matrix
    matrix[0:3, 3] = [px, py, pz]  # set translation
    return matrix

def ros_to_open3d(ros_cloud: PointCloud2) -> open3d.geometry.PointCloud:
    """Convert a ROS point cloud to an Open3D point cloud."""

    # 1st - convert ROS point cloud to numpy array
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)
    xyz = np.zeros((cloud_array.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = cloud_array["x"]
    xyz[:, 1] = cloud_array["y"]
    xyz[:, 2] = cloud_array["z"]

    # 2nd - remove invalid (NaN) points
    xyz = xyz[~np.isnan(xyz).any(axis=1)]

    # 3rd - convert to Open3D cloud
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(xyz)
    return pc


class ICPNode:
    def __init__(self):
        rospy.init_node("icp_localization", anonymous=True)
        
        # Parameters
        self.lidar_topic_name = rospy.get_param("~LIDAR_TOPIC", None)
        self.publish_topic_name = rospy.get_param("~PUBLISH_TOPIC", None)
        self.map_topic_name = rospy.get_param("~MAP_TOPIC", None)
        
        self.map_frame = rospy.get_param("~MAP_FRAME", None)
        self.base_frame = rospy.get_param("~BASE_FRAME", None)
        
        if self.lidar_topic_name is None:
            rospy.logerr("Parameter '~LIDAR_TOPIC' is required.")
            raise SystemExit(1)
        
        if self.publish_topic_name is None:
            rospy.logerr("Parameter '~PUBLISH_TOPIC' is required.")
            raise SystemExit(1)
        
        if self.map_topic_name is None:
            rospy.logerr("Parameter '~MAP_TOPIC' is required.")
            raise SystemExit(1)
        
        if self.map_frame is None:
            rospy.logerr("Parameter '~MAP_FRAME' is required.")
            raise SystemExit(1)
        
        if self.base_frame is None:
            rospy.logerr("Parameter '~BASE_FRAME' is required.")
            raise SystemExit(1)
        
        self.robot_namespace = rospy.get_namespace()
        self.reference_point_cloud: typing.Optional[PointCloud2] = None
        self.reference_point_cloud_stamp = None
        self.load_initial_guess()

        # Create a new publisher for ICP results
        self.icp_results_publisher = rospy.Publisher(self.robot_namespace + self.publish_topic_name, PoseWithCovarianceStamped, queue_size=10)
        rospy.loginfo(f"Created new publisher '{self.publish_topic_name}'")

        # Subscribe to map topic
        rospy.Subscriber(self.map_topic_name, PointCloud2, self.reference_point_cloud_callback, queue_size=1)
        rospy.loginfo(f"Subscribed to reference point cloud topic: {self.map_topic_name}")

        # Subscribe to scan topic
        rospy.Subscriber(self.lidar_topic_name, PointCloud2, self.point_cloud_callback, queue_size=10)
        rospy.loginfo(f"Subscribed to point cloud topic: {self.lidar_topic_name}")

    def load_initial_guess(self):
        # Position
        px = rospy.get_param(self.robot_namespace + "fiducial_calibration/position/x")
        py = rospy.get_param(self.robot_namespace + "fiducial_calibration/position/y")
        pz = rospy.get_param(self.robot_namespace + "fiducial_calibration/position/z")

        # Orientation (quaternion)
        ox = rospy.get_param(self.robot_namespace + "fiducial_calibration/orientation/x")
        oy = rospy.get_param(self.robot_namespace + "fiducial_calibration/orientation/y")
        oz = rospy.get_param(self.robot_namespace + "fiducial_calibration/orientation/z")
        ow = rospy.get_param(self.robot_namespace + "fiducial_calibration/orientation/w")

        self._previous_transform  = position_and_quaternion_to_matrix(px, py, pz, ox, oy, oz, ow)
        self.is_initial_guess = True

    def reference_point_cloud_callback(self, msg: PointCloud2):
        if self.reference_point_cloud_stamp is None or msg.header.stamp.to_time() != self.reference_point_cloud_stamp:
            self.reference_point_cloud = ros_to_open3d(msg)
            self.reference_point_cloud_stamp = msg.header.stamp.to_time()
            rospy.loginfo("Updated reference point cloud")

    def point_cloud_callback(self, msg: PointCloud2):
        rospy.loginfo("Received point cloud scan.")

        if self.reference_point_cloud is None:
            rospy.logwarn("Skipped ICP because no reference point cloud was published")
            return

        scan_pcd = ros_to_open3d(msg)

        if scan_pcd.is_empty():
            rospy.logwarn("Received an empty point cloud.")
            return

        # Apply ICP
        threshold = 1.0  # distance threshold
        icp_result = open3d.pipelines.registration.registration_icp(
            scan_pcd,
            self.reference_point_cloud,
            threshold,
            self._previous_transform,
            open3d.pipelines.registration.TransformationEstimationPointToPoint()
        )

        angle_change = angle_diff(icp_result.transformation, self._previous_transform)
        rospy.loginfo(f"ICP Fitness: {icp_result.fitness}")
        rospy.loginfo(f"Angle change: {angle_change}")

        # if icp_result.fitness < 0.8 and not initial_guess:
        #     rospy.logwarn("Omitting ICP result due to poor matching")
        #     return
        
        if angle_change > 0.1 and not self.is_initial_guess:
            rospy.logwarn("Omitting ICP result due to large angle change")
            return
        
        self._previous_transform = icp_result.transformation
        self.is_initial_guess = False

        message = transformation_to_pose(
            icp_result.transformation,
            self.map_frame,
        )

        self.icp_results_publisher.publish(message)
        rospy.loginfo("Published odometry")


def main():
    node = ICPNode()
    
    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
