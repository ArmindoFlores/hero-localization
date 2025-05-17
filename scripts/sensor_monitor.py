#!/usr/bin/env python3

import re
import sys
import typing

import numpy as np
import yaml
import rospy
import std_msgs.msg
from hero_localization.msg import ICPReport
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped


def compute_metric(value, t, k = 1):
    if value <= t:
        return 1
    return 1 / (k * ((value - t) / t) + 1)

class SensorMonitorNode:
    def __init__(self, node_name="sensor_monitor"):
        rospy.init_node(node_name, anonymous=True)

        self.sensor_config_file = rospy.get_param("~SENSOR_CONFIG_FILE", None)
        self.last_good_estimate = None
        self.passthrough = rospy.get_param("~PASSTHROUGH", False)

        if self.sensor_config_file is None:
            rospy.logerr("Parameter '~SENSOR_CONFIG_FILE' is required.")
            raise SystemExit(1)
        
        self.sensors = {
            "imu": [],
            "lidar": [],
            "gps": [],
            "odom": []
        }
        self.load_config()

        for i, sensor_info in enumerate(self.sensors["imu"]):
            sensor_name = sensor_info["name"]
            rospy.loginfo(f"Subscribing to IMU measurement at '{sensor_name}'")
            rospy.Subscriber(sensor_name, Imu, self.imu_callback, (i,), queue_size=100)
            self.sensors["imu"][i]["passthrough_publisher"] = rospy.Publisher(f"{sensor_name}/monitored", Imu, queue_size=100)
            self.sensors["imu"][i]["score_publisher"] = rospy.Publisher(f"{sensor_name}/plot", PointStamped, queue_size=100)

        for i, sensor_info in enumerate(self.sensors["lidar"]):
            sensor_name = sensor_info["name"]
            rospy.loginfo(f"Subscribing to LiDAR measurement at '{sensor_name}'")
            rospy.Subscriber(sensor_name, ICPReport, self.lidar_callback, (i,), queue_size=10)
            self.sensors["lidar"][i]["passthrough_publisher"] = rospy.Publisher(f"{sensor_name}/monitored", PoseWithCovarianceStamped, queue_size=10)
            self.sensors["lidar"][i]["score_publisher"] = rospy.Publisher(f"{sensor_name}/plot", PointStamped, queue_size=10)

        for i, sensor_info in enumerate(self.sensors["gps"]):
            sensor_name = sensor_info["name"]
            rospy.loginfo(f"Subscribing to GPS measurement at '{sensor_name}'")
            rospy.Subscriber(sensor_name, PoseWithCovarianceStamped, self.gps_callback, (i,), queue_size=10)
            self.sensors["gps"][i]["passthrough_publisher"] = rospy.Publisher(f"{sensor_name}/monitored", PoseWithCovarianceStamped, queue_size=10)
            self.sensors["gps"][i]["score_publisher"] = rospy.Publisher(f"{sensor_name}/plot", PointStamped, queue_size=10)

        for i, sensor_info in enumerate(self.sensors["odom"]):
            sensor_name = sensor_info["name"]
            rospy.loginfo(f"Subscribing to odometry measurement at '{sensor_name}'")
            rospy.Subscriber(sensor_name, Odometry, self.odom_callback, (i,), queue_size=30)
            self.sensors["odom"][i]["passthrough_publisher"] = rospy.Publisher(f"{sensor_name}/monitored", Odometry, queue_size=30)
            self.sensors["odom"][i]["score_publisher"] = rospy.Publisher(f"{sensor_name}/plot", PointStamped, queue_size=30)

    def compute_imu_score(self, msg: Imu, idx: int):
        angular_velocity = abs(msg.angular_velocity.z)
        angular_velocity_metric = compute_metric(angular_velocity, 0.25, 1)
        return angular_velocity_metric

    def compute_lidar_score(self, msg: ICPReport, idx: int):
        fitness_metric = 1 - msg.fitness
        rmse = msg.r2
        return compute_metric(rmse, 0.5) * compute_metric(fitness_metric, 0.5)
    
    def compute_odom_score(self, msg: Odometry, idx: int):
        linear_velocity = np.array([msg.twist.twist.linear.x, msg.twist.twist.linear.y])
        angular_velocity = abs(msg.twist.twist.angular.z)

        linear_velocity_mag = np.linalg.norm(linear_velocity)
        linear_velocity_metric = compute_metric(linear_velocity_mag, 1, 1)
        angular_velocity_metric = compute_metric(angular_velocity, 0.25, 1)
        return linear_velocity_metric * angular_velocity_metric

    def imu_callback(self, msg: Imu, args: typing.Tuple[int]):
        idx, = args
        score = self.compute_imu_score(msg, idx) if not self.passthrough else 1
        if score > 0.75:
            self.sensors["imu"][idx]["passthrough_publisher"].publish(msg)
        point = PointStamped()
        point.point.x = score
        self.sensors["imu"][idx]["score_publisher"].publish(point)

    def odom_callback(self, msg: Odometry, args: typing.Tuple[int]):
        idx, = args
        score = self.compute_odom_score(msg, idx) if not self.passthrough else 1
        if score > 0.75:
            self.sensors["odom"][idx]["passthrough_publisher"].publish(msg)
        point = PointStamped()
        point.point.x = score
        self.sensors["odom"][idx]["score_publisher"].publish(point)

    def lidar_callback(self, msg: ICPReport, args: typing.Tuple[int]):
        idx, = args
        score = self.compute_lidar_score(msg, idx) if not self.passthrough else 1
        if score > 0.75:
            pose = msg.pose
            self.last_good_estimate = pose
        elif self.last_good_estimate is not None:
            pose = self.last_good_estimate
            pose.pose.covariance = [x*10 for x in pose.pose.covariance]
        else:
            pose = None
        if pose is not None:
            self.sensors["lidar"][idx]["passthrough_publisher"].publish(msg.pose)
        point = PointStamped()
        point.point.x = score
        self.sensors["lidar"][idx]["score_publisher"].publish(point)

    def gps_callback(self, msg, args: typing.Tuple[int]):
        raise NotImplementedError

    def load_config(self):
        with open (self.sensor_config_file, "r") as f:
            config = yaml.safe_load(f)

        namespace = rospy.get_namespace()
        topics = dict([(topic.replace(namespace, "", 1), type) for topic, type in rospy.get_published_topics(namespace)])

        KEY_REGEX = re.compile(r"^((?:pose)|(?:odom)|(?:imu))(\d+)$")
        for key, value in config.items():
            if isinstance(value, str) and not value.endswith("/monitored"):
                continue
            match = KEY_REGEX.match(key)
            if match is None:
                continue
            message_type = match.group(1)
            sensor_type = None
            if value in topics:
                sensor_type = self.sensor_type_from_message_type(topics[value])
            if sensor_type is None:
                sensor_type = "lidar" if message_type == "pose" else message_type
            self.sensors[sensor_type].append({
                "name": value[:-10],
                "passthrough_publisher": None,
                "score_publisher": None,
                "data": [],
            })

    @staticmethod
    def sensor_type_from_message_type(message_type: str):
        if message_type == "sensor_msgs/Imu":
            return "imu"
        elif message_type == "nav_msgs/Odometry":
            return "odom"
        elif message_type == "hero_localization/ICPReport":
            return "lidar"
        return None


def main():
    node = SensorMonitorNode()

    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass