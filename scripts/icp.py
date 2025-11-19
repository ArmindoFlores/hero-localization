#!/usr/bin/env python3

import sys
import time
import typing

import numpy as np
import open3d
import ros_numpy
import rospy
import tf.transformations as tf_trans
import tf2_ros
from nav_msgs.msg import Odometry
from std_srvs.srv import Trigger, TriggerResponse
from hero_localization.msg import ICPReport
from geometry_msgs.msg import PoseWithCovarianceStamped
from sensor_msgs.msg import PointCloud2


def angle_diff(m1, m2):
    q1 = tf_trans.quaternion_from_matrix(m1)
    q2 = tf_trans.quaternion_from_matrix(m2)
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

    matrix = tf_trans.quaternion_matrix(rotation)
    matrix[0:3, 3] = translation
    return matrix

def transformation_to_pose(transform, frame_id, time: rospy.Time = None):
    """Convert a 4x4 transformation matrix (from Open3D ICP) to a ROS Pose message."""

    # Extract translation
    translation = transform[0:3, 3]

    # Convert rotation to quaternion
    quaternion = tf_trans.quaternion_from_matrix(transform)

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
        5e-3, 0, 0, 0, 0, 0,
        0, 5e-3, 0, 0, 0, 0,
        0, 0, 5e-3, 0, 0, 0,
        0, 0, 0, 5e-2, 0, 0,
        0, 0, 0, 0, 5e-2, 0,
        0, 0, 0, 0, 0, 5e-2
    ]
    # pose.pose.covariance = [
    #     1e-4, 0, 0, 0, 0, 0,
    #     0, 1e-4, 0, 0, 0, 0,
    #     0, 0, 1e6, 0, 0, 0,
    #     0, 0, 0, 1e6, 0, 0,
    #     0, 0, 0, 0, 1e6, 0,
    #     0, 0, 0, 0, 0, 1e-3
    # ]

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
    distances = np.linalg.norm(xyz, axis=1)
    xyz = xyz[distances <= R]

    # 4th - convert to Open3D cloud
    pc = open3d.geometry.PointCloud()
    pc.points = open3d.utility.Vector3dVector(xyz)
    return pc


class ICPNode:
    def __init__(self, node_name="icp_localization"):
        rospy.init_node(node_name, anonymous=True)
        
        # Parameters
        self.scan_max_age = 200
        self.do_global_search = True
        self.lidar_topic_name = rospy.get_param("~LIDAR_TOPIC", None)
        self.publish_topic_name = rospy.get_param("~PUBLISH_TOPIC", None)
        self.map_topic_name = rospy.get_param("~MAP_TOPIC", None)
        self.global_map_topic_name = rospy.get_param("~GLOBAL_MAP_TOPIC", None)
        self.include_icp_results = rospy.get_param("~INCLUDE_ICP_RESULTS", True)
        self.map_frame = rospy.get_param("~MAP_FRAME", None)
        self.base_frame = rospy.get_param("~BASE_FRAME", None)
        self.global_estimate_topic = rospy.get_param("~GLOBAL_ESTIMATE", None)
        self.adaptive_distance_threshold = 2
        self.adaptive_max_iterations = 30
        self.last_global_estimate = None
        self.last_global_estimate_tf = None
        self.previous_used_global_transform_time = rospy.get_time()
        
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
        
        if self.global_estimate_topic is None:
            rospy.logerr("Parameter '~GLOBAL_ESTIMATE' is required.")
            raise SystemExit(1)
        
        self.buffer = tf2_ros.Buffer(rospy.Duration(secs=10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        
        self.robot_namespace = rospy.get_namespace()
        self.reference_point_cloud: typing.Optional[PointCloud2] = None
        self.global_point_cloud: typing.Optional[PointCloud2] = None
        self.global_fpfh = None
        self.reference_point_cloud_stamp = None
        self.load_initial_guess()
        self.last_registration_timestamp = None

        # Create a new publisher for ICP results
        if not self.include_icp_results:
            self.icp_results_publisher = rospy.Publisher(self.robot_namespace + self.publish_topic_name, PoseWithCovarianceStamped, queue_size=10)
        else:
            self.icp_results_publisher = rospy.Publisher(self.robot_namespace + self.publish_topic_name, ICPReport, queue_size=10)
        rospy.loginfo(f"Created new publisher '{self.publish_topic_name}'")

        # Subscribe to global estimate
        rospy.Subscriber(self.global_estimate_topic, Odometry, self.global_estimate_callback, queue_size=10)

        # Subscribe to map topic
        rospy.Subscriber(self.map_topic_name, PointCloud2, self.reference_point_cloud_callback, queue_size=1)
        rospy.loginfo(f"Subscribed to reference point cloud topic: {self.map_topic_name}")

        if self.global_map_topic_name is not None:
            # Subscribe to map topic
            rospy.Subscriber(self.global_map_topic_name, PointCloud2, self.global_point_cloud_callback, queue_size=1)
            rospy.loginfo(f"Subscribed to global point cloud topic: {self.global_map_topic_name}")

        # Subscribe to scan topic
        rospy.Subscriber(self.lidar_topic_name, PointCloud2, self.point_cloud_callback, queue_size=10)
        rospy.loginfo(f"Subscribed to point cloud topic: {self.lidar_topic_name}")

        rospy.Service("trigger_global_search", Trigger, self.handle_global_search_trigger)
        rospy.loginfo("Created new service 'trigger_global_search'")

    def transform_lidar_to_base_link(self, pcd, timestamp, source_frame: str):
        transform = self.buffer.lookup_transform(
            target_frame=self.base_frame,
            source_frame=source_frame,
            time=timestamp,
        )
        trans = transform.transform.translation
        rot = transform.transform.rotation
        T = tf_trans.quaternion_matrix([rot.x, rot.y, rot.z, rot.w])
        T[:3, 3] = [trans.x, trans.y, trans.z]
        return pcd.transform(T)

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

        self.previous_transform  = position_and_quaternion_to_matrix(px, py, pz, ox, oy, oz, ow)
        self.previous_transform_time = rospy.get_time()
        self.is_initial_guess = True

    def handle_global_search_trigger(self, _):
        self.do_global_search = True
        return TriggerResponse(
            success=True,
            message="ICP global search successfully triggered."
        )

    def reference_point_cloud_callback(self, msg: PointCloud2):
        if self.reference_point_cloud_stamp is None or msg.header.stamp.to_time() != self.reference_point_cloud_stamp:
            self.reference_point_cloud = ros_to_open3d(msg, 1000)
            # self.reference_point_cloud.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
            self.reference_point_cloud_stamp = msg.header.stamp.to_time()
            rospy.loginfo("Updated reference point cloud")

    def global_point_cloud_callback(self, msg: PointCloud2):
        if self.global_point_cloud is None:
            self.global_point_cloud = ros_to_open3d(msg, 1000).voxel_down_sample(0.5)
            self.global_point_cloud.estimate_normals(
                search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=1.0, max_nn=30)
            )
            self.global_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
                self.global_point_cloud,
                search_param=open3d.geometry.KDTreeSearchParamHybrid(
                    radius=2.5, max_nn=100)
            )
            rospy.loginfo("Updated global point cloud")

    def global_estimate_callback(self, msg):
        self.last_global_estimate = msg
        pos = msg.pose.pose.position
        translation = [pos.x, pos.y, pos.z]
        ori = msg.pose.pose.orientation
        quaternion = [ori.x, ori.y, ori.z, ori.w]
        self.last_global_estimate_tf = position_and_quaternion_to_matrix(
            *translation,
            *quaternion
        )

    def point_cloud_callback(self, msg: PointCloud2):
        start_time = time.perf_counter()
        how_old = (rospy.get_time() - msg.header.stamp.to_time()) * 1000
        if how_old > self.scan_max_age:
            rospy.logwarn(f"Discarded scan because it was too old ({how_old} ms > {self.scan_max_age} ms)")
            return

        if self.do_global_search and self.global_point_cloud is None:
            self.do_global_search = False
            rospy.logwarn("Doing local instead of global search because no global map was published")

        if self.do_global_search:
            rospy.loginfo("Performing global search")
        
        reference_point_cloud = self.global_point_cloud if self.do_global_search else self.reference_point_cloud
        if reference_point_cloud is None:
            rospy.logwarn("Skipped ICP because no reference point cloud was published")
            return

        scan_pcd = ros_to_open3d(msg)
        if scan_pcd.is_empty():
            rospy.logwarn("Received an empty point cloud.")
            return
        
        scan_pcd = scan_pcd.voxel_down_sample(0.1)
        scan_pcd = self.transform_lidar_to_base_link(scan_pcd, msg.header.stamp, msg.header.frame_id)

        cov = np.sum(np.abs(self.last_global_estimate.pose.covariance)) if self.last_global_estimate is not None else 10000
        previous_transform_age = rospy.get_time() - self.previous_transform_time
        last_used_global_transform_age = rospy.get_time() - self.previous_used_global_transform_time
        
        if cov < 1 or self.previous_transform is None or previous_transform_age > 15 or last_used_global_transform_age > 15:
            self.previous_used_global_transform_time = rospy.get_time()
            initial_guess = self.last_global_estimate_tf
        else:
            initial_guess = self.previous_transform

        if initial_guess is None:
            rospy.logwarn_throttle(1, "Couldn't perform ICP because no initial estimate was available")
            return

        if self.do_global_search:
            # If lost, perform global search before ICP
            self.do_global_search = False
            rospy.loginfo("Computing FPFH features")
            scan_pcd.estimate_normals(search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30))
            scan_fpfh = open3d.pipelines.registration.compute_fpfh_feature(
                scan_pcd,
                search_param=open3d.geometry.KDTreeSearchParamHybrid(radius=2, max_nn=30)
            )
            distance_threshold = 2
            rospy.loginfo("Performing registration")
            result = open3d.pipelines.registration.registration_ransac_based_on_feature_matching(
                self.global_point_cloud, scan_pcd,
                self.global_fpfh, scan_fpfh, True,
                distance_threshold,
                open3d.pipelines.registration.TransformationEstimationPointToPoint(False),
                4,  # RANSAC correspondence set size
                [
                    open3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
                    open3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(distance_threshold)
                ],
                open3d.pipelines.registration.RANSACConvergenceCriteria(2000000, 500)
            )
            initial_guess = result.transformation
            rospy.loginfo("Done")

        # Apply ICP
        distance_threshold = self.adaptive_distance_threshold
        max_iter = self.adaptive_max_iterations
        icp_result = open3d.pipelines.registration.registration_icp(
            scan_pcd,
            self.reference_point_cloud,
            self.adaptive_distance_threshold,
            initial_guess,
            open3d.pipelines.registration.TransformationEstimationPointToPoint(),
            open3d.pipelines.registration.ICPConvergenceCriteria(
                max_iteration=int(max_iter),
            ),
        )
        current_time = rospy.get_time()
        elapsed = 0 if self.last_registration_timestamp is None else current_time - self.last_registration_timestamp
        self.last_registration_timestamp = current_time
        k = elapsed / 0.1

        r2 = icp_result.inlier_rmse / distance_threshold
        if icp_result.fitness > 0.7 and r2 < 0.5:
            self.adaptive_max_iterations = min(self.adaptive_max_iterations, max(15, self.adaptive_max_iterations - 0.05 * k))
            self.adaptive_distance_threshold = min(self.adaptive_distance_threshold, max(0.1, self.adaptive_distance_threshold - 0.0125 * k))
        if icp_result.fitness < 0.5 or r2 > 0.5:
        # if icp_result.fitness < 0.3 or r2 > 0.7:
            self.adaptive_max_iterations = max(self.adaptive_max_iterations, min(100, self.adaptive_max_iterations + 0.05 * k))
            self.adaptive_distance_threshold = max(self.adaptive_distance_threshold, min(2, self.adaptive_distance_threshold + 0.0125 * k))

        self.previous_transform = icp_result.transformation
        self.previous_transform_time = rospy.get_time()
        self.is_initial_guess = False

        transformation = icp_result.transformation

        pose_message = transformation_to_pose(
            transformation,
            self.map_frame,
            msg.header.stamp
        )

        end_time = time.perf_counter()
        elapsed = end_time - start_time

        if self.include_icp_results:
            message = ICPReport(
                pose=pose_message,
                r2=r2,
                fitness=icp_result.fitness,
                elapsed=elapsed,
                distance_threshold=distance_threshold,
                max_iter=max_iter
            )
        else:
            message = pose_message

        rospy.loginfo(f"ICP: {elapsed*1000:.2f} ms")
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
