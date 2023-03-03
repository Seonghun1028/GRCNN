#! /usr/bin/env python

# import os
# import datetime
# import moveit_msgs.msg
# from std_msgs.msg import Int16, String
# from geometry_msgs.msg import Twist, Pose
# from franka_msgs.msg import FrankaState, Errors as FrankaErrors
# from moveit_commander.conversions import pose_to_list, list_to_pose
# import math as m
# from math import pi
# from sre_constants import SUCCESS


import rospy
import sys
import time
import moveit_commander
import copy
import utils
import geometry_msgs.msg
import franka_gripper.msg
import actionlib

from tf2_ros import StaticTransformBroadcaster, TransformBroadcaster
from geometry_msgs.msg import TransformStamped, Pose
from visualization_msgs.msg import Marker, MarkerArray
from tf.transformations import *


#from msg import Grasp #경로 수정
from grcnn.srv import GraspPrediction #경로 수정

def pose_to_tf(frame_id, child_frame_id, pose):
    t = TransformStamped()
    t.header.frame_id = frame_id
    t.child_frame_id = child_frame_id
    if isinstance(pose, Pose):
        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation
    else:
        rospy.logerr("pose is not Pose type")
    return t


def marker_msg(pose, stamp, i):
    marker_ = Marker()
    marker_.id = i
    marker_.header.frame_id = "/world"
    marker_.header.stamp = stamp
    marker_.type = marker_.SPHERE
    marker_.action = marker_.ADD
    marker_.pose = pose
    marker_.scale.x = 0.05
    marker_.scale.y = 0.05
    marker_.scale.z = 0.05
    marker_.color.r = 1
    marker_.color.g = 0
    marker_.color.b = 0
    marker_.color.a = 1
    return marker_

def marker_array_msg(waypoints):
    stamp = rospy.Time.now()
    marker_array = MarkerArray()
    for i, pose in enumerate(waypoints):
        marker_array.markers.append(marker_msg(pose, stamp, i))
    return marker_array

class PandaOpenLoopGraspController():
    """
    Perform open-loop grasps from a single viewpoint using the Panda robot.
    """
    def __init__(self):
        moveit_commander.roscpp_initialize(sys.argv)
        rospy.init_node('panda_grasp')

        # Get group_commander from MoveGroupCommander
        self.robot = moveit_commander.RobotCommander()
        self.scene = moveit_commander.PlanningSceneInterface()
        self.group_name = "panda_arm"
        self.move_group = moveit_commander.MoveGroupCommander(self.group_name)
        self.move_group.set_planner_id("RRTConnectkConfigDefault")
        self.move_group.set_end_effector_link('panda_link8')

        # Planning Parameter
        self.planning_frame = self.move_group.get_planning_frame()
        self.group_names = self.robot.get_group_names()
        self.eef_link = self.move_group.get_end_effector_link()
        self.grasp_name = "panda_hand"
        self.grasp_state = False
        self.hand_group = moveit_commander.MoveGroupCommander(self.grasp_name)

        # Joint
        self.joint = [moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint1')]
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint2'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint3'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint4'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint5'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint6'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_joint7'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_finger_joint1'))
        self.joint.append(moveit_commander.RobotCommander.Joint(
            self.robot, 'panda_finger_joint2'))

        # Visualization
        self.br = StaticTransformBroadcaster()
        self.marker_array_pub = rospy.Publisher('waypoints', MarkerArray, queue_size=1)


    



    def linear_motion(self, distance, avoid_collision=True, reference="world"):
        move_group = self.move_group
        waypoints = []
        wpose = move_group.get_current_pose().pose

        if reference == "world":
            wpose.position.x += distance[0]
            wpose.position.y += distance[1]
            wpose.position.z += distance[2]
        elif reference == "eef":
            lpose = distance + [0, 0, 0, 1]
            wpose = utils.concatenate_to_pose(wpose, lpose)
        waypoints.append(copy.deepcopy(wpose))

        # Visualize
        self.marker_array_pub.publish(marker_array_msg(waypoints))
        
        # Plan
        start = time.time()
        (plan, fraction) = move_group.compute_cartesian_path(waypoints,
                                                                0.03,
                                                                0.0,
                                                                avoid_collisions=avoid_collision)
        planning_time = time.time() - start
        
        # if self.self_play:
        #     rospy.loginfo('linear_motion: {}'.format(fraction))
        # else:
        #     input('linear_motion: {}'.format(fraction))
        
        # Execute
        start = time.time()
        move_group.execute(plan, wait=True)
        execution_time = time.time() - start

        # info
        mp_info = dict()
        mp_info['planning_time'] = planning_time
        mp_info['execution_time'] = execution_time
        mp_info['fraction'] = fraction
        mp_info['success'] = True if fraction>0 else False

        return plan, mp_info








    def get_grasp_pose(self):
        rospy.wait_for_service('/predict')
        grcnn_srv = rospy.ServiceProxy('/predict', GraspPrediction) #grasp정보 받음
        ret = grcnn_srv()
        if not ret.success:
            return False
        print(ret.best_grasp)
        self.pose = ret.best_grasp
        return self.pose

    def move_to(self, pose_goal):
        self.move_group.set_pose_target(pose_goal)
        plan = self.move_group.go(wait=True)
        self.move_group.stop()
        self.move_group.clear_pose_targets()

    def grasp_initialize(self,grasp_size=0.04):
        try:
            hand_group = self.hand_group 
            hand_group.set_joint_value_target([grasp_size, 0])
            hand_group.go()
        except Exception as e:
            rospy.logerr(e)
    
    def pose_initialize(self):
        pose_goal = geometry_msgs.msg.Pose()
        cur_pose = self.move_group.get_current_pose().pose

        pose_goal.position.x = cur_pose.position.x
        pose_goal.position.y = cur_pose.position.y
        pose_goal.position.z = 0.78

        # quat = quaternion_from_euler(pi, 0, pi/4)

        pose_goal.orientation.x = 0.9238795
        pose_goal.orientation.y = -0.3826834
        pose_goal.orientation.z = 0
        pose_goal.orientation.w = 0

        self.move_to(pose_goal)
    
    def grasp(self, width):
        client = actionlib.SimpleActionClient('/franka_gripper/grasp', franka_gripper.msg.GraspAction)
        client.wait_for_server()
        goal = franka_gripper.msg.GraspGoal()
        goal.width = width
        goal.epsilon.inner = 0.04
        goal.epsilon.outer = 0.04
        goal.speed = 0.1
        goal.force = 5
        client.sent_goal(goal)
        client.wait_for_result()
        
        # return client.get_result()   # if successs or if fail 등으로 응용
    
    def move_cartesian(self,x,y,z):
        waypoints = []
        cur_pose = self.move_group.get_current_pose().pose
        wpose = cur_pose

        wpose.position.x = x
        waypoints.append(copy.deepcopy(wpose))

        wpose.position.y = y
        waypoints.append(copy.deepcopy(wpose))



    

if __name__ == '__main__':
    # initialize
    move_group = moveit_commander.MoveGroupCommander("panda_arm")
    pose_goal = geometry_msgs.msg.Pose()
    pose_test = PandaOpenLoopGraspController()
    pose_test.grasp_initialize()
    pose_test.pose_initialize()

    input()

    # get grasp_pose from server
    target_pose = pose_test.get_grasp_pose()

    input()

    # Cartesian path(x,y)
    waypoints = []
    cur_pose = move_group.get_current_pose().pose
    wpose = cur_pose
    
    wpose.position.x = target_pose.position.x
    waypoints.append(copy.deepcopy(wpose))
    wpose.position.y = target_pose.position.y
    waypoints.append(copy.deepcopy(wpose))

    (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
    move_group.execute(plan, wait=True)

    input()

    # Change orientation of end effector
    cur_pose = move_group.get_current_pose().pose
    pose_goal.orientation = target_pose.orientation

    move_group.set_pose_target(pose_goal)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    input()

    # Cartesian path(z)
    waypoints = []
    cur_pose = move_group.get_current_pose().pose
    wpose = cur_pose

    wpose.position.z = target_pose.position.z
    waypoints.append(copy.deepcopy(wpose))
    
    (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.01, 0.0)
    move_group.execute(plan, wait=True)

    input()

    # Try close gripper
    pose_test.grasp(width=0.02)

    input()

    # Pick up
    pose_test.linear_motion([0, 0, 0.05], True)

    input()

    # Place
    pose_goal = geometry_msgs.msg.Pose()
    pose_goal.position.x = 0.2015
    pose_goal.position.y = 0.2733
    pose_goal.position.z = 0.3700
    pose_goal.orientation.x = 0.9238795
    pose_goal.orientation.y = -0.3826834
    pose_goal.orientation.z = 0
    pose_goal.orientation.w = 0
    
    pose_test.move_to(pose_goal)

    input()

    # Try open gripper
    pose_test.grasp(width=0.04)

    input()

    # Up
    pose_test.linear_motion([0, 0, 0.05], True)

