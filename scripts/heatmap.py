#!/usr/bin/env python3
import io
import pickle
import re
import sys
import yaml

import tf2_ros
import numpy as np
import rospy
from geometry_msgs.msg import PointStamped
from hero_localization.msg import Heatmap, IntTuple


class HeatMapNode:
    def __init__(self, node_name="heatmap"):
        rospy.init_node(node_name, anonymous=True)
        self.heatmap = {}
        self.save_file = rospy.get_param("~SAVE_FILE", None)
        self.granularity = rospy.get_param("~GRANULARITY", 0.5)
        self.recency_bias = rospy.get_param("~RECENCY_BIAS", 0.1)
        self.sensor_config_file = rospy.get_param("~SENSOR_CONFIG_FILE", None)
        self.map_frame = rospy.get_param("~MAP_FRAME", None)
        self.base_frame = rospy.get_param("~BASE_FRAME", None)
        self.fga = rospy.get_param("~FGA", True)
        self.initial_map_file = rospy.get_param("~INITIAL_MAP_FILE", None)

        if self.save_file is None:
            rospy.logerr("Parameter '~SAVE_FILE' is required.")
            raise SystemExit(1)
        
        if self.map_frame is None:
            rospy.logerr("Parameter '~MAP_FRAME' is required.")
            raise SystemExit(1)
        
        if self.base_frame is None:
            rospy.logerr("Parameter '~BASE_FRAME' is required.")
            raise SystemExit(1)

        if self.sensor_config_file is None:
            rospy.logerr("Parameter '~SENSOR_CONFIG_FILE' is required.")
            raise SystemExit(1)
        
        self.buffer = tf2_ros.Buffer(rospy.Duration(secs=10))
        self.listener = tf2_ros.TransformListener(self.buffer)
        self.load_config()

        self.publishers = {}

        for topic in self.heatmap:
            rospy.Subscriber(topic, PointStamped, self.on_measurement_received, (topic,), queue_size=100)
            self.publishers[topic] = rospy.Publisher(topic[:-4]+"heatmap", Heatmap, queue_size=10)
            rospy.loginfo(f"Subscribing to sensor score at '{topic}'")

        rospy.Timer(rospy.Duration(5), self.save_results)
        rospy.Timer(rospy.Duration(1), self.publish_results)

    def publish_results(self, _):
        for topic, publisher in self.publishers.items():
            heatmap_copy = self.heatmap[topic].copy()

            hm = Heatmap(
                size=len(heatmap_copy.keys()),
                coordinates=list([IntTuple(first=int(key[0]), second=int(key[1])) for key in heatmap_copy.keys()]),
                values=list(heatmap_copy.values()),
                granularity=self.granularity
            )
            publisher.publish(hm)

    def on_measurement_received(self, msg: PointStamped, args):
        try:
            current_transform = self.buffer.lookup_transform(
                target_frame=self.map_frame,
                source_frame=self.base_frame,
                time=rospy.Time(0),
            )
        except Exception as e:
            rospy.logwarn_throttle(1, f"Couldn't retrieve global estimate ({str(e)})")
            return
        trans = current_transform.transform.translation
        if self.fga:
            coords = np.array([trans.x, trans.y])
        else:
            coords = np.array([trans.x, trans.y, trans.z])
        self.register_measurement(args[0], coords, msg.point.x)

    def save_results(self, _):
        try:
            with open(self.save_file, "wb") as f:
                pickle.dump(self.heatmap, f)
        except Exception as e:
            rospy.logerr(f"Error saving heatmap: {str(e)}")

    def register_measurement(self, topic, coords, value):
        key = self.as_key(coords)
        current_value = self.heatmap[topic].get(key, None)
        if current_value is None:
            self.heatmap[topic][key] = value
        else:
            self.heatmap[topic][key] = self.combine_values(current_value, value)
            
    def combine_values(self, old_value, new_value):
        old_w = 1 / (1 + self.recency_bias)
        new_w = self.recency_bias / (1 + self.recency_bias)
        return old_value * old_w + new_value * new_w

    def as_key(self, coords):
        return tuple(self.quantize(coords))

    def quantize(self, coords):
        return np.floor(coords / self.granularity)
    
    def load_config(self):
        if self.initial_map_file is None:
            with open (self.sensor_config_file, "r") as f:
                config = yaml.safe_load(f)

            KEY_REGEX = re.compile(r"^((?:pose)|(?:odom)|(?:imu))(\d+)$")
            for key, value in config.items():
                if not isinstance(value, str):
                    continue
                if not value.endswith("/monitored"):
                    if "odometry/gps" in value:
                        value = value.replace("odometry/", "", 1) + "/monitored"
                    else:
                        continue

                match = KEY_REGEX.match(key)
                if match is None:
                    continue
                self.heatmap[value[:-9]+"plot"] = {}
        else:
            with open(self.initial_map_file, "rb") as f:
                self.heatmap = pickle.load(f)


def main():
    node = HeatMapNode()
    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
