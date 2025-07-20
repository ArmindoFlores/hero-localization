#!/usr/bin/env python3
import sys

import rospy


class BaselineNode:
    def __init__(self, node_name="baseline"):
        rospy.init_node(node_name, anonymous=True)
        rospy.Timer(rospy.Duration(1), self.do_nothing)

    def do_nothing(self, _):
        rospy.logdebug("Baseline doing nothing")

def main():
    node = BaselineNode()
    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
