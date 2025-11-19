#!/usr/bin/env python3

import dataclasses
import re
import sys
import typing

import numpy as np
from scipy.signal import butter, filtfilt
import yaml
import cv2
from cv_bridge import CvBridge
import rospy
from tf.transformations import quaternion_matrix
from scipy.spatial.transform import Rotation as R
from std_srvs.srv import Trigger
from hero_localization.msg import ICPReport, Heatmap
from nav_msgs.msg import Odometry
from sensor_msgs.msg import Imu, Image, NavSatFix
from geometry_msgs.msg import PoseWithCovarianceStamped, PointStamped


ALPHA = 0.5
COMPUTE_JUMPS_FOR_LIDAR = False

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
    if normalized_cutoff > 1:
        normalized_cutoff = 0.9
    if normalized_cutoff < 0:
        normalized_cutoff = 0.1

    b, a = butter(N=order, Wn=normalized_cutoff, btype="low", analog=False)
    x_filtered = filtfilt(b, a, x)

    noise = x - x_filtered
    energy = np.sum(noise ** 2)
    return energy

def nearest_index(array, value):
    return np.abs(array - value).argmin()

def lla_to_ecef(lat, lon, alt):
    # lat, lon in degrees, alt in meters
    R = 6371000.0  # mean Earth radius (m)
    lat_rad = np.radians(lat)
    lon_rad = np.radians(lon)
    x = (R + alt) * np.cos(lat_rad) * np.cos(lon_rad)
    y = (R + alt) * np.cos(lat_rad) * np.sin(lon_rad)
    z = (R + alt) * np.sin(lat_rad)
    return x, y, z

@dataclasses.dataclass
class SensorData:
    name: str
    data_buffer_t: typing.List[float]
    data_buffer_x: list
    active: bool
    heatmap: typing.Dict[typing.Tuple[int, int], float]
    heatmap_granularity: float
    heatmap_plot: rospy.Publisher = None
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
        self.use_heatmap = rospy.get_param("~USE_HEATMAP", False)
        self.map_frame = rospy.get_param("~MAP_FRAME", None)
        self.base_frame = rospy.get_param("~BASE_FRAME", None)
        self.global_estimate_topic = rospy.get_param("~GLOBAL_ESTIMATE", None)
        self.use_spatial_interpolation = rospy.get_param("~USE_SPATIAL_INTERPOLATION", False)
        self.last_global_estimate = None

        if self.map_frame is None:
            rospy.logerr("Parameter '~MAP_FRAME' is required.")
            raise SystemExit(1)
        
        if self.base_frame is None:
            rospy.logerr("Parameter '~BASE_FRAME' is required.")
            raise SystemExit(1)

        if self.sensor_config_file is None:
            rospy.logerr("Parameter '~SENSOR_CONFIG_FILE' is required.")
            raise SystemExit(1)
        
        if self.global_estimate_topic is None:
            rospy.logerr("Parameter '~GLOBAL_ESTIMATE' is required.")
            raise SystemExit(1)
        
        self.sensors: typing.Dict[SensorType, typing.List[SensorData]] = {
            "imu": [],
            "lidar": [],
            "gps": [],
            "odom": []
        }
        
        rospy.Subscriber(self.global_estimate_topic, Odometry, self.global_estimate_callback, queue_size=10)
        self.load_config()

        for i, sensor_info in enumerate(self.sensors["imu"]):
            rospy.loginfo(f"Subscribing to IMU measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, Imu, self.imu_callback, (i,), queue_size=100)
            self.sensors["imu"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", Imu, queue_size=100)
            self.sensors["imu"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=100)
            if self.use_heatmap:
                rospy.Subscriber(f"{sensor_info.name}/heatmap", Heatmap, self.heatmap_callback, ("imu", i,), queue_size=2)
                sensor_info.heatmap_plot = rospy.Publisher(f"{sensor_info.name}/heatmap/plot", Image, queue_size=1)

        for i, sensor_info in enumerate(self.sensors["lidar"]):
            rospy.loginfo(f"Subscribing to LiDAR measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, ICPReport, self.lidar_callback, (i,), queue_size=10)
            self.sensors["lidar"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", PoseWithCovarianceStamped, queue_size=10)
            self.sensors["lidar"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=10)
            if self.use_heatmap:
                rospy.Subscriber(f"{sensor_info.name}/heatmap", Heatmap, self.heatmap_callback, ("lidar", i,), queue_size=2)
                sensor_info.heatmap_plot = rospy.Publisher(f"{sensor_info.name}/heatmap/plot", Image, queue_size=1)

        for i, sensor_info in enumerate(self.sensors["gps"]):
            rospy.loginfo(f"Subscribing to GPS measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, NavSatFix, self.gps_callback, (i,), queue_size=10)
            self.sensors["gps"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", NavSatFix, queue_size=10)
            self.sensors["gps"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=10)
            if self.use_heatmap:
                rospy.Subscriber(f"{sensor_info.name}/heatmap", Heatmap, self.heatmap_callback, ("gps", i,), queue_size=2)
                sensor_info.heatmap_plot = rospy.Publisher(f"{sensor_info.name}/heatmap/plot", Image, queue_size=1)

        for i, sensor_info in enumerate(self.sensors["odom"]):
            rospy.loginfo(f"Subscribing to odometry measurement at '{sensor_info.name}'")
            rospy.Subscriber(sensor_info.name, Odometry, self.odom_callback, (i,), queue_size=30)
            self.sensors["odom"][i].passthrough_publisher = rospy.Publisher(f"{sensor_info.name}/monitored", Odometry, queue_size=30)
            self.sensors["odom"][i].score_publisher = rospy.Publisher(f"{sensor_info.name}/plot", PointStamped, queue_size=30)
            if self.use_heatmap:
                rospy.Subscriber(f"{sensor_info.name}/heatmap", Heatmap, self.heatmap_callback, ("odom", i,), queue_size=2)
                sensor_info.heatmap_plot = rospy.Publisher(f"{sensor_info.name}/heatmap/plot", Image, queue_size=1)

        if self.icp_reset_trigger is not None:
            rospy.loginfo("Waiting for ICP reset trigger service...")
            rospy.wait_for_service(self.icp_reset_trigger)
            self.icp_reset_trigger_service = rospy.ServiceProxy(self.icp_reset_trigger, Trigger)
            rospy.loginfo("Found ICP reset trigger service")

        if self.use_heatmap:
            rospy.Timer(rospy.Duration(1), self.publish_heatmap_plots)

    @staticmethod
    def add_to_buffer(buffer: typing.List, data, max_size = 200):
        if len(buffer) >= max_size:
            buffer.pop(0)
        buffer.append(data)

    def buffered_data_x(self, sensor: str, idx: int):
        return self.sensors[sensor][idx].data_buffer_x
    
    def buffered_data_t(self, sensor: str, idx: int):
        return self.sensors[sensor][idx].data_buffer_t
    
    def get_latest_position_estimate(self):
        try:
            if self.last_global_estimate is None:
                rospy.logwarn_throttle(1, f"Couldn't retrieve global estimate")
                return
            position = self.last_global_estimate.pose.pose.position
            return np.array([position.x, position.y, position.z])
        except Exception as e:
            rospy.logwarn_throttle(1, f"Couldn't retrieve global estimate ({str(e)})")
            return None
        
    def get_latest_velocity_estimate(self):
        try:
            if self.last_global_estimate is None:
                rospy.logwarn_throttle(1, f"Couldn't retrieve global velocity estimate")
                return
            
            odom = self.last_global_estimate
            velocity_local = odom.twist.twist.linear
            q = odom.pose.pose.orientation

            quat = [q.x, q.y, q.z, q.w]
            rot_matrix = quaternion_matrix(quat)[:3, :3]

            vel_local = np.array([velocity_local.x, velocity_local.y, velocity_local.z])
            vel_global = rot_matrix.dot(vel_local)

            return vel_global
        except Exception as e:
            rospy.logwarn_throttle(1, f"Couldn't retrieve global velocity estimate ({str(e)})")
            return None
        
    def global_estimate_callback(self, msg):
        self.last_global_estimate = msg

    def publish_heatmap_plots(self, _):
        for sensor_type in self.sensors:
            for sensor_info in self.sensors[sensor_type]:
                xs = [x for x, _ in sensor_info.heatmap.keys()]
                ys = [y for _, y in sensor_info.heatmap.keys()]
                if len(xs) < 1 or len(ys) < 1: continue

                x_max, x_min = max(xs), min(xs)
                y_max, y_min = max(ys), min(ys)

                # Create grid
                width = int(x_max - x_min + 1)
                height = int(y_max - y_min + 1)
                Z = np.zeros((height, width), dtype=np.float32)

                for (center_x, center_y), mean_value in sensor_info.heatmap.items():
                    xi = int(center_x - x_min)
                    yi = int(center_y - y_min)
                    if 0 <= xi < width and 0 <= yi < height:
                        Z[yi, xi] = 1 - mean_value

                # Normalize values between 0–255
                Z_norm = cv2.normalize(Z, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

                scale = 20  # each grid cell = 20 pixels
                heatmap = cv2.applyColorMap(Z_norm, cv2.COLORMAP_VIRIDIS)
                heatmap = cv2.resize(
                    heatmap,
                    (width*scale, height*scale),
                    interpolation=cv2.INTER_NEAREST
                )

                position = self.get_latest_position_estimate()
                velocity = self.get_latest_velocity_estimate()
                next_position = position + 1.5*velocity
                radius = scale // 3
                center = (
                    int((position[0] - x_min) * scale + scale / 2),
                    int((position[1] - y_min) * scale + scale / 2),
                )
                cv2.circle(heatmap, center, radius, (255, 255, 255), -1)
                center = (
                    int((next_position[0] - x_min) * scale + scale / 2),
                    int((next_position[1] - y_min) * scale + scale / 2),
                )
                cv2.circle(heatmap, center, radius, (128, 128, 128), -1)

                bridge = CvBridge()
                msg = bridge.cv2_to_imgmsg(heatmap, encoding="bgr8")
                sensor_info.heatmap_plot.publish(msg)

    def heatmap_callback(self, msg: Heatmap, args):
        sensor_type, idx = args
        new_heatmap = {}
        for i in range(msg.size):
            coords = msg.coordinates
            new_heatmap[(coords[i].first, coords[i].second)] = msg.values[i]
        self.sensors[sensor_type][idx].heatmap = new_heatmap
        self.sensors[sensor_type][idx].heatmap_granularity = msg.granularity

    @staticmethod
    def _8_neighborhood(pos):
        neighborhood = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if i == 0 and j == 0: continue
                neighborhood.append((pos[0] + i, pos[1] + j))
        return neighborhood
    
    def _get_spatial_interpolation_weight(self, position, base_position, sensor):
        if sensor.heatmap.get(position) is None:
            return 0
        return np.sqrt((sensor.heatmap_granularity*position[0]-base_position[0])**2 + (sensor.heatmap_granularity*position[1]-base_position[1])**2)
    
    def _normalize_spatial_interpolation_weights(self, samples):
        total_weight = sum([weight for _, weight in samples])
        if total_weight == 0:
            return samples
        return [(position, weight / total_weight) for position, weight in samples]
    
    def _do_spatial_interpolation(self, samples, heatmap):
        return sum([weight * heatmap.get(position) if weight > 0 else 0 for position, weight in samples])

    def _heatmap_get(self, sensor, base_position):
        center_position = tuple(np.floor(base_position / sensor.heatmap_granularity))
        if self.use_spatial_interpolation:
            positions = [
                *self._8_neighborhood(center_position), 
                center_position
            ]
            samples = [
                (position, self._get_spatial_interpolation_weight(position, base_position, sensor))
                for position in positions
            ]
            normalized_samples = self._normalize_spatial_interpolation_weights(samples)
            value = self._do_spatial_interpolation(normalized_samples, sensor.heatmap)
            return value
        return sensor.heatmap.get(center_position)

    def get_heatmap_component(self, sensor_type: SensorType, idx: int):
        this_sensor = self.sensors[sensor_type][idx]
        position = self.get_latest_position_estimate()
        velocity = self.get_latest_velocity_estimate()
        
        if len(this_sensor.heatmap) == 0:
            return None
        
        if position is None or velocity is None:
            return None
        
        # Using flat ground assumption
        if len(next(iter(this_sensor.heatmap.keys()))) == 2:
            position = position[:-1]
            velocity = velocity[:-1]
        
        # FIXME: doesn't work in 3D
        # coords = np.floor(position / this_sensor.heatmap_granularity)
        # value = this_sensor.heatmap.get(tuple(coords))

        # if velocity is not None:
        #     next_position = np.floor(position / this_sensor.heatmap_granularity + np.sign(velocity))
        #     value_at_next_position = this_sensor.heatmap.get(tuple(next_position))
        #     if value is None:
        #         value = value_at_next_position
        #     elif value_at_next_position is not None:
        #         distance_to_next_position = np.linalg.norm(next_position * this_sensor.heatmap_granularity - position)
        #         m = np.sqrt(2) * this_sensor.heatmap_granularity
        #         a = distance_to_next_position / m
        #         b = (m - distance_to_next_position) / m
        #         value = a * value + b * value_at_next_position

        return self._heatmap_get(this_sensor, position + 1.5*velocity)

    def compute_imu_score(self, idx: int):
        timed_out = False
        last_time = self.sensors["imu"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated IMU {idx} data due to timeout")
            timed_out = True
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

        score = angular_velocity_metric * noise_metric
        return 0 if timed_out else score, score 

    def compute_lidar_score(self, msg: ICPReport, idx: int):
        timed_out = False
        last_time = self.sensors["lidar"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated LiDAR {idx} data due to timeout")
            timed_out = True
        fitness_metric = 1 - msg.fitness
        rmse = msg.r2

        t = self.buffered_data_t("lidar", idx)
        data = self.buffered_data_x("lidar", idx)

        if COMPUTE_JUMPS_FOR_LIDAR:
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
        # score = compute_metric(rmse, 0.4) * compute_metric(fitness_metric, 0.4, 2) * velocity_metric * angular_velocity_metric
        score = compute_metric(rmse, 0.5) * compute_metric(fitness_metric, 0.5, 1) * velocity_metric * angular_velocity_metric
        return 0 if timed_out else score, score
    
    def compute_odom_score(self, idx: int):
        timed_out = False
        last_time = self.sensors["odom"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated odometry {idx} data due to timeout")
            timed_out = True
        t = self.buffered_data_t("odom", idx)
        data = self.buffered_data_x("odom", idx)
        linear_velocity = np.array(data[-1][:3])
        angular_velocity_mag = np.linalg.norm(data[-1][3:])

        linear_velocity_mag = np.linalg.norm(linear_velocity)
        # For all:
        # linear_velocity_metric = compute_metric(linear_velocity_mag, 1.3, 1)
        # For Turflynx:
        linear_velocity_metric = compute_metric(linear_velocity_mag, 2, 1)
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

        score = linear_velocity_metric * angular_velocity_metric * noise_metric
        return 0 if timed_out else score, score
    
    def compute_gps_score(self, idx: int):
        timed_out = False
        last_time = self.sensors["gps"][idx].last_published_at
        if last_time is not None and rospy.get_time() - last_time > 1:
            rospy.logwarn_throttle(1, f"Invalidated gps {idx} data due to timeout")
            timed_out = True
        t = self.buffered_data_t("gps", idx)
        data = self.buffered_data_x("gps", idx)
        
        if len(data) < 2:
            return 0 if timed_out else 1, 1
        
        x_now, y_now, z_now = lla_to_ecef(*data[-1][:3])
        x_before, y_before, z_before = lla_to_ecef(*data[-2][:3])
        error = data[-1][3]

        linear_velocity_mag = np.linalg.norm(np.array([
            x_now - x_before,
            y_now - y_before,
            z_now - z_before
        ])) / (t[-1] - t[-2])
        
        # For all:
        # linear_velocity_metric = compute_metric(linear_velocity_mag, 1.3, 1)
        # For Turflynx:
        linear_velocity_metric = compute_metric(linear_velocity_mag, 2.5, 1)
        error_metric = compute_metric(error, 0.5, 2)

        score = linear_velocity_metric * error_metric
        return 0 if timed_out else score, score

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
        pre_heat_score, pre_heat_score_without_timeout = self.compute_imu_score(idx) if not self.passthrough else (1, 1)
        score = pre_heat_score

        if self.use_heatmap:
            hm_score = self.get_heatmap_component("imu", idx)
            if hm_score is not None:
                score = ALPHA*score + (1-ALPHA)*hm_score

        self.sensors["imu"][idx].last_published_at = rospy.get_time()
        if score <= 0.75:
            self.sensors["imu"][idx].active = False
        if score > 0.75:
            self.sensors["imu"][idx].passthrough_publisher.publish(msg)
            self.sensors["imu"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = pre_heat_score_without_timeout
        point.point.y = score
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
        pre_heat_score, pre_heat_score_without_timeout = self.compute_odom_score(idx) if not self.passthrough else (1, 1)
        score = pre_heat_score

        if self.use_heatmap:
            hm_score = self.get_heatmap_component("odom", idx)
            if hm_score is not None:
                score = ALPHA*score + (1-ALPHA)*hm_score

        self.sensors["odom"][idx].last_published_at = rospy.get_time()
        if score <= 0.75:
            self.sensors["odom"][idx].active = False
        if score > 0.75:
            self.sensors["odom"][idx].passthrough_publisher.publish(msg)
            self.sensors["odom"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = pre_heat_score_without_timeout
        point.point.y = score
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
        pre_heat_score, pre_heat_score_without_timeout = self.compute_lidar_score(msg, idx) if not self.passthrough else (1, 1)
        score = pre_heat_score

        if self.use_heatmap:
            hm_score = self.get_heatmap_component("lidar", idx)
            if hm_score is not None:
                score = ALPHA*score + (1-ALPHA)*hm_score

        self.sensors["lidar"][idx].last_published_at = rospy.get_time()
        pose = msg.pose
        if score <= 0.75:
            pose.pose.covariance = [x*10 for x in pose.pose.covariance]
            self.sensors["lidar"][idx].active = False
            if self.icp_reset_trigger is not None and not self.sensors["lidar"][idx].active:
                rospy.logwarn("Asking ICP node to perform a global search")
                self.icp_reset_trigger_service()
        if score > 0.75:
            self.last_good_estimate = pose
            self.sensors["lidar"][idx].passthrough_publisher.publish(pose)
            self.sensors["lidar"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = pre_heat_score_without_timeout
        point.point.y = score
        point.point.z = msg.fitness
        self.sensors["lidar"][idx].score_publisher.publish(point)

    def gps_callback(self, msg, args: typing.Tuple[int]):
        idx, = args
        self.add_to_buffer(
            self.buffered_data_t("gps", idx), 
            msg.header.stamp.to_sec()
        )
        self.add_to_buffer(
            self.buffered_data_x("gps", idx), 
            (
                msg.latitude,
                msg.longitude,
                msg.altitude,
                np.sqrt(np.trace(np.array(msg.position_covariance).reshape(3, 3))),
            )
        )
        pre_heat_score, pre_heat_score_without_timeout = self.compute_gps_score(idx) if not self.passthrough else (1, 1)
        score = pre_heat_score

        if self.use_heatmap:
            hm_score = self.get_heatmap_component("gps", idx)
            if hm_score is not None:
                score = ALPHA*score + (1-ALPHA)*hm_score

        self.sensors["gps"][idx].last_published_at = rospy.get_time()
        if score <= 0.75:
            self.sensors["gps"][idx].active = False
        if score > 0.75:
            self.sensors["gps"][idx].passthrough_publisher.publish(msg)
            self.sensors["gps"][idx].active = True
        point = PointStamped()
        point.header.stamp = rospy.Time()
        point.point.x = pre_heat_score_without_timeout
        point.point.y = score
        self.sensors["gps"][idx].score_publisher.publish(point)

    def load_config(self):
        with open (self.sensor_config_file, "r") as f:
            config = yaml.safe_load(f)

        namespace = rospy.get_namespace()
        topics = dict([(topic.replace(namespace, "", 1), type) for topic, type in rospy.get_published_topics(namespace)])

        KEY_REGEX = re.compile(r"^((?:pose)|(?:odom)|(?:imu))(\d+)$")
        for key, value in config.items():
            if not isinstance(value, str):
                continue
            if not value.endswith("/monitored"):
                if "odometry/gps" in value: # Hack
                    sensor_type = "gps"
                    value = value.replace("odometry/gps", "gps", 1) + "/monitored"
                else:
                    continue
            else:
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
                active=True,
                heatmap={},
                heatmap_granularity=1,
            ))

    @staticmethod
    def sensor_type_from_message_type(message_type: str) -> typing.Optional[SensorType]:
        if message_type == "sensor_msgs/Imu":
            return "imu"
        elif message_type == "nav_msgs/Odometry":
            return "odom"
        elif message_type == "hero_localization/ICPReport":
            return "lidar"
        elif message_type == "sensor_msgs/NavSatFix":
            return "gps"
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