#!/usr/bin/env python3

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, Point, Quaternion
from std_srvs.srv import Empty
import time
import rosbag
import subprocess
import os
from time import sleep
import numpy as np

#import gym


class DigitalTwinEnv():
    '''Open AI gym environment for the Ship Digital Twin'''

    def __init__(self):
        '''Constructor for the DigitalTwin Environment'''


        self.result = Float64()
        self.result.data = 0
        self.x_offset = 220
        self.y_offset = 40
        self.maxSimulations = np.inf
        self.maxTime = np.inf



        self.build_action_space()


        self.pub = rospy.Publisher('move_usv/goal', Odometry, queue_size=10)
        rospy.init_node('patrol')
        self.rate = rospy.Rate(1) # 10h
        rospy.Subscriber("move_usv/result", Float64, self.get_result)
        rospy.wait_for_service('/gazebo/unpause_physics')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/reset_simulation')
        self.unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self.pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self.resetSimulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)
        #unpause()

        simulationNumber = 1
        while not rospy.is_shutdown():    
            try:
                rospy.logerr("Simulation number %d", simulationNumber)
                for pose in self.generate_random_waypoints(3):
                    goal = self.goal_pose(pose)
                    self.pub.publish(goal)
                    self.rate.sleep()
                    if (rospy.get_time() > self.maxTime):
                        break;
                    while self.result.data == 0.0:
                        self.pub.publish(goal)
                        self.rate.sleep()
                        if (rospy.get_time() > self.maxTime):
                            break;
                
                simulationNumber = simulationNumber + 1
                rospy.logerr("Increasing simulationNumber. now: %d", simulationNumber)
                if (simulationNumber > self.maxSimulations):
                    rospy.logerr("All simulations have been done. Pausing gazebo")
                    self.pause()
                else:
                    rospy.logerr("preparing new simulation!")
                    #rospy.logerr("pause simulation!")
                    self.pause()
                    #rospy.logerr("wait!")
                    time.sleep(1)
                    #rospy.logerr("reset simulation!")
                    self.resetSimulation()
                    #rospy.logerr("wait!")   
                    time.sleep(1)
                    #rospy.logerr("start new simulation!")
                    self.unpause()
                    rospy.logerr("Continue simulation!") 
            except rospy.ROSInterruptException:
                rospy.logerr("ROS InterruptException! Just ignore the exception!") 
            except rospy.ROSTimeMovedBackwardsException:
                rospy.logerr("ROS Time Backwards! Just ignore the exception!")

    
    def build_action_space(self):
        '''Build the Open AI Gym action space'''

        #print("hello")
        pass
    

    def generate_random_waypoints(self, n):
        '''Generate a list of random waypoints
        
        Args:
            n: number of waypoints to randomly generate
        
        '''

        self.waypoints = [
            [(80.0, 70.0, 0.0), (0.0, 0.0, 0.7, 0.7)], 
            [(30.0, 50.0, 0.0), (0.0, 0.0, 0.7, 0.7)], 
            [(0.0, 50.0, 0.0), (0.0, 0.0, 0.7, 0.7)], 
            [(0.0, 0.0, 0.0), (0.0, 0.0, 0.7, 0.7)], 
        ]

        return self.waypoints

    

    def goal_pose(self, pose):
        goal_pose = Odometry()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = 'world'
        goal_pose.pose.pose.position = Point(pose[0][0]+self.x_offset, pose[0][1]+self.y_offset, 0.)
        return goal_pose

    def get_result(self, result_aux):
        self.result.data = result_aux.data

