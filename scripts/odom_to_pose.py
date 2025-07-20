#!/usr/bin/env python3

import sys

import rospy
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseWithCovarianceStamped

class OdometryToPoseNode:
    def __init__(self, node_name="odom_to_pose"):
        rospy.init_node(node_name, anonymous=True)
        
        # Parameters
        self.odometry_topic = rospy.get_param("~ODOM_TOPIC", None)
        self.pose_topic = rospy.get_param("~POSE_TOPIC", None)
        self.robot_namespace = rospy.get_namespace()

        if self.odometry_topic is None:
            rospy.logerr("Parameter '~ODOM_TOPIC' is required.")
            raise SystemExit(1)
        
        if self.pose_topic is None:
            rospy.logerr("Parameter '~POSE_TOPIC' is required.")
            raise SystemExit(1)

        self.pose_publisher = rospy.Publisher(self.robot_namespace + self.pose_topic, PoseWithCovarianceStamped, queue_size=10)

        rospy.Subscriber(self.odometry_topic, Odometry, self.odometry_topic_callback, queue_size=1)
        rospy.loginfo(f"Subscribed to EKF topic: {self.odometry_topic}")

    def odometry_topic_callback(self, msg: Odometry):
        pose = msg.pose
        pose_with_cov_stamped = PoseWithCovarianceStamped(
            header=msg.header,
            pose=pose
        )
        self.pose_publisher.publish(pose_with_cov_stamped)


def main():
    node = OdometryToPoseNode()
    
    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
