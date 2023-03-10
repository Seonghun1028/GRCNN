#! /usr/bin/env python

import rospy
import sys
import time
import moveit_commander
import copy
import utils
import geometry_msgs.msg
import franka_gripper.msg
import actionlib
from tf.transformations import *

#from msg import Grasp #경로 수정
from grcnn.srv import GraspPrediction #경로 수정


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
        self.move_group.set_end_effector_link('panda_link8')

        self.grasp_name = "panda_hand"
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


    def get_grasp_pose(self):
        rospy.wait_for_service('/predict')
        grcnn_srv = rospy.ServiceProxy('/predict', GraspPrediction) #grasp정보 받음
        ret = grcnn_srv()
        if not ret.success:
            return False
        print(ret.best_grasp)
        pose = ret.best_grasp
        return pose

    def move_to(self, pose_goal):
        move_group = self.move_group
        move_group.set_planner_id("RRTConnect")
        move_group.set_pose_target(pose_goal)
        plan = move_group.go(wait=True)
        move_group.stop()
        move_group.clear_pose_targets()

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
        client.send_goal(goal)
        client.wait_for_result()
        
        # return client.get_result()   # if successs or if fail 등으로 응용
    
    def move_cartesian_xy(self,x,y):
        move_group = self.move_group
        waypoints = []
        cur_pose = move_group.get_current_pose().pose
        wpose = cur_pose

        wpose.position.x = x
        waypoints.append(copy.deepcopy(wpose))
        wpose.position.y = y
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.005, 0.0)
        move_group.execute(plan, wait=True)
    
    def move_cartesian_z(self,z):
        move_group = self.move_group
        waypoints = []
        cur_pose = move_group.get_current_pose().pose
        wpose = cur_pose

        wpose.position.z = z
        waypoints.append(copy.deepcopy(wpose))

        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.005, 0.0)
        move_group.execute(plan, wait=True)

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

        (plan, fraction) = move_group.compute_cartesian_path(waypoints, 0.005, 0.0)
        move_group.execute(plan, wait=True)



if __name__ == '__main__':
    # init
    pose_test = PandaOpenLoopGraspController()
    pose_test.grasp_initialize()
    pose_test.pose_initialize()

    # get grasp_pose from server
    target_pose = pose_test.get_grasp_pose()
    
    input()

    # Cartesian path(x,y)
    pose_test.move_cartesian_xy(target_pose.position.x,target_pose.position.y)

    input()

    # Change orientation of end effector
    pose_goal = geometry_msgs.msg.Pose()
    move_group = pose_test.move_group
    cur_pose = move_group.get_current_pose().pose
    pose_goal.position = cur_pose.position
    pose_goal.orientation = target_pose.orientation

    move_group.set_pose_target(pose_goal)
    plan = move_group.go(wait=True)
    move_group.stop()
    move_group.clear_pose_targets()

    input()

    # Cartesian path(z)
    pose_test.move_cartesian_z(target_pose.position.z)

    input()

    # Try close gripper
    pose_test.grasp(width=0.02)

    input()

    # Pick up
    pose_test.linear_motion([0, 0, 0.05], True)

    input()

    # Place(Cartesian)
    pose_test.move_cartesian_z(0.6634)
    pose_test.move_cartesian_xy(0.5037,-0.3631)

    # Place(OMPL)
    # pose_goal = geometry_msgs.msg.Pose()
    # pose_goal.position.x = 0.5037
    # pose_goal.position.y = -0.3631
    # pose_goal.position.z = 0.6634
    # pose_goal.orientation.x = 0.9238795
    # pose_goal.orientation.y = -0.3826834
    # pose_goal.orientation.z = 0
    # pose_goal.orientation.w = 0    
    # pose_test.move_to(pose_goal)

    input()

    # Try open gripper
    pose_test.grasp(width=0.04)

    input()

    # Up
    pose_test.linear_motion([0, 0, 0.05], True)