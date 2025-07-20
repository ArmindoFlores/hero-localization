#!/usr/bin/env python3

import pickle
import re
import sys

import yaml
import rospy
from hero_localization.msg import ICPReport
from geometry_msgs.msg import PoseWithCovarianceStamped
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu


class TimingAnalysisNode:
    def __init__(self, node_name="timing_analisys"):
        rospy.init_node(node_name, anonymous=True)

        self.sensor_config_file = rospy.get_param("~SENSOR_CONFIG_FILE", None)
        self.save_file = rospy.get_param("~SAVE_FILE", None)

        if self.sensor_config_file is None:
            rospy.logerr("Parameter '~SENSOR_CONFIG_FILE' is required.")
            raise SystemExit(1)
        
        if self.save_file is None:
            rospy.logerr("Parameter '~SAVE_FILE' is required.")
            raise SystemExit(1)
        
        self.sensors = {}
        self.load_config()

        for sensor in self.sensors:
            if self.sensors[sensor]["type"] == "imu":
                rospy.Subscriber(sensor, Imu, self.imu_callback, (sensor,), queue_size=100)
            elif self.sensors[sensor]["type"] == "odom":
                rospy.Subscriber(sensor, Odometry, self.odom_callback, (sensor,), queue_size=30)
            elif self.sensors[sensor]["type"] == "lidar":
                # rospy.Subscriber(sensor, PoseWithCovarianceStamped, self.lidar_callback, (sensor,), queue_size=10)
                rospy.Subscriber(sensor, ICPReport, self.lidar_callback, (sensor,), queue_size=10)

        rospy.Timer(rospy.Duration(5), self.save_results)

    def save_results(self, _):
        rospy.loginfo("Saving timing results...")
        with open(self.save_file, "wb") as f:
            pickle.dump(self.sensors, f)

    def imu_callback(self, msg, sensor):
        t = rospy.get_time()
        self.sensors[sensor[0]]["data"].append((t, t - msg.header.stamp.to_sec()))

    def odom_callback(self, msg, sensor):
        t = rospy.get_time()
        self.sensors[sensor[0]]["data"].append((t, t - msg.header.stamp.to_sec()))

    def lidar_callback(self, msg, sensor):
        t = rospy.get_time()
        # self.sensors[sensor[0]]["data"].append((t, t - msg.header.stamp.to_sec()))
        self.sensors[sensor[0]]["data"].append((t, t - msg.pose.header.stamp.to_sec()))

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
            self.sensors[value[:-10]] = {"type": sensor_type, "data": []}

def main():
    node = TimingAnalysisNode()

    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass