#!/usr/bin/env python3

import dataclasses
import re
import sys
import typing

import numpy as np
from scipy.signal import butter, filtfilt
import yaml
import rospy
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
from hero_localization.msg import ICPReport
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped


def compute_metric(value, t, k = 1):
    if value <= t:
        return 1
    return 1 / (k * ((value - t) / t) + 1)

def noise_energy(t: np.ndarray, x: np.ndarray, cutoff_hz: float = 5.0, order: int = 4) -> float:
    if len(t) != len(x):
        raise ValueError("Arrays t and x must have the same length.")
    
    dt = np.mean(np.diff(t))
    fs = 1.0 / dt
    nyquist = 0.5 * fs
    normalized_cutoff = cutoff_hz / nyquist

    b, a = butter(N=order, Wn=normalized_cutoff, btype="low", analog=False)
    x_filtered = filtfilt(b, a, x)

    noise = x - x_filtered
    energy = np.sum(noise ** 2)
    return energy

@dataclasses.dataclass
class SensorData:
    name: str
    data_buffer_t: typing.List[float]
    data_buffer_x: list
    active: bool
    passthrough_publisher: typing.Optional[rospy.Publisher] = None
    score_publisher: typing.Optional[rospy.Publisher] = None
    last_published_at: typing.Optional[float] = None

SensorType = typing.Literal["imu", "lidar", "gps", "odom"]

class SensorMonitorNode:
    def __init__(self, node_name="sensor_monitor"):
        rospy.init_node(node_name, anonymous=True)

        self.sensor_config_file = rospy.get_param("~SENSOR_CONFIG_FILE", None)
        self.last_good_estimate = None
        self.passthrough = rospy.get_param("~PASSTHROUGH", False)
        self.icp_reset_trigger = rospy.get_param("~ICP_RESET_TRIGGER", None)

        if self.sensor_config_file is None:
            rospy.logerr("Parameter '~SENSOR_CONFIG_FILE' is required.")
            raise SystemExit(1)
        
        self.sensors: typing.Dict[SensorType, typing.List[SensorData]] = {
            "imu": [],
            "lidar": [],
            "gps": [],
            "odom": []
        }
        self.load_config()

        for i, sensor_info in enumerate(self.sensors["imu"]):
            rospy.loginfo(f"Subscribing to IMU measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, Imu, self.imu_callback, (i,), queue_size=100)
            self.sensors["imu"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", Imu, queue_size=100)
            self.sensors["imu"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=100)

        for i, sensor_info in enumerate(self.sensors["lidar"]):
            rospy.loginfo(f"Subscribing to LiDAR measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, ICPReport, self.lidar_callback, (i,), queue_size=10)
            self.sensors["lidar"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", PoseWithCovarianceStamped, queue_size=10)
            self.sensors["lidar"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=10)

        for i, sensor_info in enumerate(self.sensors["gps"]):
            rospy.loginfo(f"Subscribing to GPS measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, PoseWithCovarianceStamped, self.gps_callback, (i,), queue_size=10)
            self.sensors["gps"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", PoseWithCovarianceStamped, queue_size=10)
            self.sensors["gps"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=10)

        for i, sensor_info in enumerate(self.sensors["odom"]):
            rospy.loginfo(f"Subscribing to odometry measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, Odometry, self.odom_callback, (i,), queue_size=30)
            self.sensors["odom"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", Odometry, queue_size=30)
            self.sensors["odom"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=30)

        if self.icp_reset_trigger is not None:
            rospy.loginfo("Waiting for ICP reset trigger service...")
            rospy.wait_for_service(self.icp_reset_trigger)
            self.icp_reset_trigger_service = rospy.ServiceProxy(self.icp_reset_trigger, Trigger)
            rospy.loginfo("Found ICP reset trigger service")

    @staticmethod
    def add_to_buffer(buffer: typing.List, data, max_size = 200):
        if len(buffer) >= max_size:
            buffer.pop(0)
        buffer.append(data)

    def buffered_data_x(self, sensor: str, idx: int):
        return self.sensors[sensor][idx].data_buffer_x
    
    def buffered_data_t(self, sensor: str, idx: int):
        return self.sensors[sensor][idx].data_buffer_t

    def compute_imu_score(self, idx: int):
        last_time = self.sensors["imu"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated IMU {idx} data due to timeout")
            return 0
        t = self.buffered_data_t("imu", idx)
        data = self.buffered_data_x("imu", idx)
        angular_velocity_norm = np.linalg.norm(data[-1])
        angular_velocity_metric = compute_metric(angular_velocity_norm, 1.5, 0.5)

        if len(data) < 100:
            noise_metric = 1
        else:
            energy = noise_energy(
                t[-100:],
                [np.linalg.norm(d) for d in data[-100:]],
                15
            )
            noise_metric = compute_metric(energy, 1, 1)

        return angular_velocity_metric * noise_metric

    def compute_lidar_score(self, msg: ICPReport, idx: int):
        last_time = self.sensors["lidar"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated LiDAR {idx} data due to timeout")
            return 0
        fitness_metric = 1 - msg.fitness
        rmse = msg.r2

        t = self.buffered_data_t("lidar", idx)
        data = self.buffered_data_x("lidar", idx)

        if False:# len(data) > 1:
            t_now = t[-1]
            t_before = t[-2]

            position_now = np.array(data[-1][:3])
            position_before = np.array(data[-2][:3])
            velocity = np.linalg.norm(position_now - position_before) / (t_now - t_before)
            velocity_metric = compute_metric(velocity, 1, 1)
            
            quat_now = np.array(data[-1][3:])
            quat_before = np.array(data[-2][3:])
            r1 = R.from_quat(quat_before)
            r2 = R.from_quat(quat_now)
            r_rel = r2 * r1.inv()
            angle_change = r_rel.magnitude() / (t_now - t_before)
            angular_velocity_metric = compute_metric(angle_change, 1, 1)

        else:
            velocity_metric = 1
            angular_velocity_metric = 1
        return compute_metric(rmse, 0.4) * compute_metric(fitness_metric, 0.4, 2) * velocity_metric * angular_velocity_metric
    
    def compute_odom_score(self, idx: int):
        last_time = self.sensors["odom"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated odometry {idx} data due to timeout")
            return 0
        t = self.buffered_data_t("odom", idx)
        data = self.buffered_data_x("odom", idx)
        linear_velocity = np.array(data[-1][:3])
        angular_velocity_mag = np.linalg.norm(data[-1][3:])

        linear_velocity_mag = np.linalg.norm(linear_velocity)
        linear_velocity_metric = compute_metric(linear_velocity_mag, 1.3, 1)
        angular_velocity_metric = compute_metric(angular_velocity_mag, 1.3, 1)

        if len(data) < 100:
            noise_metric = 1
        else:
            energy = noise_energy(
                t[-100:],
                [np.linalg.norm(d[3:]) for d in data[-100:]],
                3
            )
            noise_metric = compute_metric(energy, 1, 1)

        return linear_velocity_metric * angular_velocity_metric * noise_metric
    
    def can_function_without(self, sensor_type: SensorType, idx: int):
        """
        This function returns True if ignoring this sensor's data
        *should* still allow for localization.
        """
        # sensors_to_evaluate = []
        # if sensor_type == "lidar":
        #     sensors_to_evaluate = ["odom", "lidar"]
        # elif sensor_type == "odom":
        #     sensors_to_evaluate = ["odom", "lidar"]
        # elif sensor_type == "imu":
        #     sensors_to_evaluate = ["imu", "odom", "lidar"]
        
        # for other_sensor_type in sensors_to_evaluate:
        #     for i, sensor_data in enumerate(self.sensors[other_sensor_type]):
        #         if other_sensor_type == sensor_type and i == idx:
        #             # Skip evaluating the current sensor
        #             continue
        #         if sensor_data.active:
        #             # Found at least one active sensor, so localization can
        #             # be performed
        #             return True
        # rospy.logwarn(f"Can't function without {sensor_type} {idx}!")
        # return False
        return True

    def imu_callback(self, msg: Imu, args: typing.Tuple[int]):
        idx, = args
        # if len(self.buffered_data_t("imu", idx)) == 200:
        #     return
        self.add_to_buffer(
            self.buffered_data_t("imu", idx), 
            msg.header.stamp.to_sec()
        )
        self.add_to_buffer(
            self.buffered_data_x("imu", idx), 
            (
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z,
            )
        )
        score = self.compute_imu_score(idx) if not self.passthrough else 1
        self.sensors["imu"][idx].last_published_at = rospy.get_time()
        if score <= 0.75:
            self.sensors["imu"][idx].active = False
        if score > 0.75 or not self.can_function_without("imu", idx):
            self.sensors["imu"][idx].passthrough_publisher.publish(msg)
            self.sensors["imu"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = score
        self.sensors["imu"][idx].score_publisher.publish(point)

    def odom_callback(self, msg: Odometry, args: typing.Tuple[int]):
        idx, = args
        self.add_to_buffer(
            self.buffered_data_t("odom", idx), 
            msg.header.stamp.to_sec()
        )
        self.add_to_buffer(
            self.buffered_data_x("odom", idx), 
            (
                msg.twist.twist.linear.x,
                msg.twist.twist.linear.y,
                msg.twist.twist.linear.z,
                msg.twist.twist.angular.x,
                msg.twist.twist.angular.y,
                msg.twist.twist.angular.z,
            )
        )
        score = self.compute_odom_score(idx) if not self.passthrough else 1
        self.sensors["odom"][idx].last_published_at = rospy.get_time()
        if score <= 0.75:
            self.sensors["odom"][idx].active = False
        if score > 0.75 or not self.can_function_without("odom", idx):
            self.sensors["odom"][idx].passthrough_publisher.publish(msg)
            self.sensors["odom"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = score
        self.sensors["odom"][idx].score_publisher.publish(point)

    def lidar_callback(self, msg: ICPReport, args: typing.Tuple[int]):
        idx, = args
        self.add_to_buffer(
            self.buffered_data_t("lidar", idx), 
            msg.pose.header.stamp.to_sec()
        )
        self.add_to_buffer(
            self.buffered_data_x("lidar", idx), 
            (
                msg.pose.pose.pose.position.x,
                msg.pose.pose.pose.position.y,
                msg.pose.pose.pose.position.z,
                msg.pose.pose.pose.orientation.x,
                msg.pose.pose.pose.orientation.y,
                msg.pose.pose.pose.orientation.z,
                msg.pose.pose.pose.orientation.w,
            )
        )
        score = self.compute_lidar_score(msg, idx) if not self.passthrough else 1
        self.sensors["lidar"][idx].last_published_at = rospy.get_time()
        pose = msg.pose
        if score <= 0.75:
            pose.pose.covariance = [x*10 for x in pose.pose.covariance]
            self.sensors["lidar"][idx].active = False
            if self.icp_reset_trigger is not None and not self.sensors["lidar"][idx].active:
                rospy.logwarn("Asking ICP node to perform a global search")
                self.icp_reset_trigger_service()
        if score > 0.75 or not self.can_function_without("lidar", idx):
            self.last_good_estimate = pose
            self.sensors["lidar"][idx].passthrough_publisher.publish(msg.pose)
            self.sensors["lidar"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = score
        point.point.y = msg.fitness
        point.point.z = msg.r2
        self.sensors["lidar"][idx].score_publisher.publish(point)

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
            self.sensors[sensor_type].append(SensorData(
                name=value[:-10],
                data_buffer_t=[],
                data_buffer_x=[],
                active=True
            ))

    @staticmethod
    def sensor_type_from_message_type(message_type: str) -> typing.Optional[SensorType]:
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