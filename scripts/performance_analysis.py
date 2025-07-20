#!/usr/bin/env python3

import pickle
import psutil
import sys
import time

import rosnode
import rospy
import xmlrpc.client


class PerformanceAnalysisNode:
    def __init__(self, node_name="performance_analysis"):
        rospy.init_node(node_name, anonymous=True)

        self.nodes = rospy.get_param("~NODES", [])
        self.rate = rospy.get_param("~RATE", 2)
        self.save_file = rospy.get_param("~SAVE_FILE", None)

        if self.save_file is None:
            rospy.logerr("Parameter '~SAVE_FILE' is required.")
            raise SystemExit(1)
        
        self.pids = {}
        self.data = {}
        self.last_save = time.time()
        for node in self.nodes:
            self.pids[node] = self.get_node_pid(node)
            self.data[node] = {
                "time": [],
                "memory": [],
                "cpu": []
            }
            if self.pids[node] is None:
                rospy.logerr(f"Failed to get the process ID of node '{node}'")
                return
            
        rospy.loginfo(f"Analyzing performance of the following nodes: {', '.join(self.nodes)}")
        rospy.Timer(rospy.Duration(1 / self.rate), self.monitor_processes)

    def monitor_processes(self, _):
        for node, process_id in self.pids.items():
            process = psutil.Process(process_id)
            self.data[node]["time"].append(time.time())
            self.data[node]["memory"].append(process.memory_percent())
            self.data[node]["cpu"].append(process.cpu_percent(0.1))

        if time.time() - self.last_save > 5:
            self.save_results()
            self.last_save = time.time()

    def save_results(self):
        rospy.loginfo("Saving performance results...")
        with open(self.save_file, "wb") as f:
            pickle.dump(self.data, f)

    def get_node_pid(self, node_name: str, timeout = 30):
        last_exc = None
        start_time = time.time()
        while time.time() - start_time < timeout:
            try:
                err, msg, node_api = rosnode.get_api_uri(rospy.get_master(), node_name, True)
                if err == -1:
                    raise RuntimeError(f"Could not find API URI for node '{node_name}': {msg}")
                proxy = xmlrpc.client.ServerProxy(node_api)
                code, msg, pid = proxy.getPid("/rosnode")
                if code != 1:
                    raise RuntimeError(f"XML-RPC call failed: {msg}")
                return pid
            except Exception as e:
                last_exc = e
            time.sleep(0.1)
        rospy.logerr(f"Failed to get PID of node '{node_name}': {last_exc}")
        return

def main():
    node = PerformanceAnalysisNode()
    rospy.spin()
    return 0

if __name__ == "__main__":
    try:
        sys.exit(main())
    except rospy.ROSInterruptException:
        pass
