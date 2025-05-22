#!/usr/bin/env python3

import sys
import time
import typing

import numpy as np
import open3d
import ros_numpy
import rospy
import tf
import tf.transformations
import tf.transformations as tf_trans
import tf2_ros
import tf2_geometry_msgs
from hero_localization.msg import ICPReport
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2


def angle_diff(m1, m2):
    q1 = tf.transformations.quaternion_from_matrix(m1)
    q2 = tf.transformations.quaternion_from_matrix(m2)
    q1_norm = q1 / np.linalg.norm(q1)
    q2_norm = q2 / np.linalg.norm(q2)
    
    dot = np.clip(np.abs(np.dot(q1_norm, q2_norm)), 0.0, 1.0)
    return 2 * np.arccos(dot)

def tf_to_matrix(transform):
    """
    Converts a geometry_msgs/TransformStamped message into a 4x4 transformation matrix.
    """
    trans = transform.transform.translation
    rot = transform.transform.rotation

    translation = np.array([trans.x, trans.y, trans.z])
    rotation = np.array([rot.x, rot.y, rot.z, rot.w])

    matrix = tf.transformations.quaternion_matrix(rotation)
    matrix[0:3, 3] = translation
    return matrix

def transformation_to_pose(transform, frame_id, time: rospy.Time = None):
    """Convert a 4x4 transformation matrix (from Open3D ICP) to a ROS Pose message."""

    # Extract translation
    translation = transform[0:3, 3]

    # Convert rotation to quaternion
    quaternion = tf.transformations.quaternion_from_matrix(transform)

    # Create Pose message
    pose = PoseWithCovarianceStamped()
    pose.header.stamp = rospy.Time.now() if time is None else time
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

def ros_to_open3d(ros_cloud: PointCloud2, R: float = 45.0) -> open3d.geometry.PointCloud:
    """Convert a ROS point cloud to an Open3D point cloud, discarding points further than R meters from the origin."""

    # 1st - convert ROS point cloud to numpy array
    cloud_array = ros_numpy.point_cloud2.pointcloud2_to_array(ros_cloud)
    xyz = np.zeros((cloud_array.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = cloud_array["x"]
    xyz[:, 1] = cloud_array["y"]
    xyz[:, 2] = cloud_array["z"]

    # 2nd - remove invalid (NaN) points
    xyz = xyz[~np.isnan(xyz).any(axis=1)]

    # 3rd - remove points further than R meters from origin
    # distances = np.linalg.norm(xyz, axis=1)
    # xyz = xyz[distances <= R]

    # 4th - convert to Open3D cloud
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(xyz)
    return pc


class ICPNode:
    def __init__(self, node_name="icp_localization"):
        rospy.init_node(node_name, anonymous=True)
        
        # Parameters
        self.scan_max_age = 200
        self.lidar_topic_name = rospy.get_param("~LIDAR_TOPIC", None)
        self.publish_topic_name = rospy.get_param("~PUBLISH_TOPIC", None)
        self.map_topic_name = rospy.get_param("~MAP_TOPIC", None)
        self.include_icp_results = rospy.get_param("~INCLUDE_ICP_RESULTS", True)
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
        
        self.buffer = tf2_ros.Buffer(rospy.Duration(secs=10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        
        self.robot_namespace = rospy.get_namespace()
        self.reference_point_cloud: typing.Optional[PointCloud2] = None
        self.reference_point_cloud_stamp = None
        self.load_initial_guess()

        # Create a new publisher for ICP results
        if not self.include_icp_results:
            self.icp_results_publisher = rospy.Publisher(self.robot_namespace + self.publish_topic_name, PoseWithCovarianceStamped, queue_size=10)
        else:
            self.icp_results_publisher = rospy.Publisher(self.robot_namespace + self.publish_topic_name, ICPReport, queue_size=10)
        rospy.loginfo(f"Created new publisher '{self.publish_topic_name}'")

        # Subscribe to map topic
        rospy.Subscriber(self.map_topic_name, PointCloud2, self.reference_point_cloud_callback, queue_size=1)
        rospy.loginfo(f"Subscribed to reference point cloud topic: {self.map_topic_name}")

        # Subscribe to scan topic
        rospy.Subscriber(self.lidar_topic_name, PointCloud2, self.point_cloud_callback, queue_size=10)
        rospy.loginfo(f"Subscribed to point cloud topic: {self.lidar_topic_name}")

    def transform_lidar_to_base_link(self, pose: PoseWithCovarianceStamped, source_frame: str):
        transform = self.buffer.lookup_transform(
            target_frame=self.base_frame,
            source_frame=source_frame,
            time=pose.header.stamp,
        )
        
        return tf2_geometry_msgs.do_transform_pose(pose.pose, transform)

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
            self.reference_point_cloud = ros_to_open3d(msg, 1000)
            # self.reference_point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            self.reference_point_cloud_stamp = msg.header.stamp.to_time()
            rospy.loginfo("Updated reference point cloud")

    def point_cloud_callback(self, msg: PointCloud2):
        t0 = time.perf_counter()
        how_old = (rospy.get_time() - msg.header.stamp.to_time()) * 1000
        if how_old > self.scan_max_age:
            rospy.logwarn(f"Discarded scan because it was too old ({how_old} ms > {self.scan_max_age} ms)")
            return

        if self.reference_point_cloud is None:
            rospy.logwarn("Skipped ICP because no reference point cloud was published")
            return

        scan_pcd = ros_to_open3d(msg)
        if scan_pcd.is_empty():
            rospy.logwarn("Received an empty point cloud.")
            return
        
        scan_pcd = scan_pcd.voxel_down_sample(0.25)
        t1 = time.perf_counter()

        # transform = self.buffer.lookup_transform(
        #     target_frame=self.map_frame,
        #     source_frame=self.base_frame,
        #     time=rospy.Time(0),
        #     # time=msg.header.stamp,
        # )
        # t_mat = tf_to_matrix(transform)

        # initial_guess = t_mat
        initial_guess = self._previous_transform
        t2 = time.perf_counter()

        # Apply ICP
        distance_threshold = 2
        icp_result = open3d.pipelines.registration.registration_icp(
            scan_pcd,
            self.reference_point_cloud,
            distance_threshold,
            initial_guess,
            open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            # open3d.pipelines.registration.TransformationEstimationPointToPlane(),
            open3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=30,
                relative_fitness=0,
                relative_rmse=1e-6,
            ),
        )
        t3 = time.perf_counter()

        # angle_change = angle_diff(icp_result.transformation, initial_guess)
        rospy.loginfo(f"ICP Fitness: {icp_result.fitness}  ICP RMSE: {icp_result.inlier_rmse / distance_threshold}")

        # if icp_result.fitness < 0.8 and not initial_guess:
        #     rospy.logwarn("Omitting ICP result due to poor matching")
        #     return
        
        # if angle_change > 0.1 and not self.is_initial_guess:
        #     rospy.logwarn("Omitting ICP result due to large angle change")
        #     return
        
        self._previous_transform = icp_result.transformation
        self.is_initial_guess = False

        pose_message = transformation_to_pose(
            icp_result.transformation,
            self.map_frame,
            msg.header.stamp
        )
        pose_message.pose.pose = self.transform_lidar_to_base_link(
            pose_message,
            msg.header.frame_id
        ).pose

        t4 = time.time()
        elapsed = t4 - t0

        if self.include_icp_results:
            message = ICPReport(
                pose=pose_message,
                r2=icp_result.inlier_rmse / distance_threshold,
                fitness=icp_result.fitness,
                elapsed=elapsed
            )
        else:
            message = pose_message

        rospy.loginfo("Timing breakdown:")
        rospy.loginfo(f"Conversion: {(t1 - t0)*1000:.2f} ms")
        rospy.loginfo(f"Transform: {(t2 - t1)*1000:.2f} ms")
        rospy.loginfo(f"ICP: {(t3 - t2)*1000:.2f} ms")
        rospy.loginfo(f"Total: {elapsed*1000:.2f} ms")
        self.icp_results_publisher.publish(message)


def main():
    node = ICPNode()
    
    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
