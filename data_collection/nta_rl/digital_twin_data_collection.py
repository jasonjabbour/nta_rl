#!/usr/bin/env python

import rospy
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64
from geometry_msgs.msg import Twist, Point, Quaternion
from sensor_msgs.msg import JointState
from std_srvs.srv import Empty
import time
import rosbag
import subprocess
from time import sleep
import numpy as np
import random
import csv
import os
import inspect

MODE = 'collect-data'
WAYPOINTS_PER_EPISODE = 10
TOTAL_EPISODES = 1
WAYPOINT_MAX_XY_COORDINATES = 50
WAYPOINT_MIN_XY_COORDINATES = 0
MAX_TIME_PER_EPISODE = 36000 # seconds
WIND_SPEED_SET = 3
SEED = 24
CURRENTDIR = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
CSV_FILE_NAME = '/digitaltwin_data_300waypoints_1ep_1'


class DigitalTwinDataCollection():
    '''Digital Twin Data Collection Class and Setting Boat Goals'''
    def __init__(self, 
                 mode=MODE):
        '''
        Constructor for Digital Twin Data Collection Class

        Args:
            mode: a string discription such as train or test
            
        '''

        if mode == 'collect-data':
            self.setup_data_collection()


    def setup_data_collection(self):
        '''Driver for collecting data'''

        # Initialize message type
        self._result = Float64()
        self._result.data = 0

        # Offset taking into account starting position of ship
        self._x_offset = 220
        self._y_offset = 0
    
        # Create topic to publish goal to
        self._pub = rospy.Publisher('move_usv/goal', Odometry, queue_size=10)

        # Initialize this node as patrol
        rospy.init_node('patrol')

        # Set the rate to publish goal
        self._rate = rospy.Rate(10) # 10h

        # Subscribe to the result topic
        rospy.Subscriber("move_usv/result", Float64, self.get_result)

        # Create all way points
        self._all_episode_waypoint_lst = self.generate_random_waypoints()

        # Counter
        self._episode_counter = 1
        self._terminate = False

        # Set seed
        self.set_seed()

        # Initialize Data Lists
        self.initialize_data_lists()

        # Subscribe to topics for data listening
        self.sub4data()

        # Gazebo services
        rospy.wait_for_service('/gazebo/unpause_physics')
        rospy.wait_for_service('/gazebo/pause_physics')
        rospy.wait_for_service('/gazebo/reset_simulation')
        self._unpause = rospy.ServiceProxy('/gazebo/unpause_physics', Empty)
        self._pause = rospy.ServiceProxy('/gazebo/pause_physics', Empty)
        self._resetSimulation = rospy.ServiceProxy('/gazebo/reset_simulation', Empty)

        # Publish Waypoints as goals for boat to reach
        self.publish_waypoints()

        # Save collected data to csv
        self.save_data_2_csv()

    
    def publish_waypoints(self):
        '''Run through all waypoints generated and publish them on topic until boat has reached the waypoint'''

        while not rospy.is_shutdown(): 
            #self._unpause()   
            try:
                # Start an episode
                while self._episode_counter <= TOTAL_EPISODES:

                    # Wait for UWSIM to Load before unpausing
                    if self._episode_counter == 1:
                        time.sleep(25)

                    # Unpause gazebo
                    self._unpause()

                    rospy.logerr("Episode number %d", self._episode_counter)

                    # Select waypoint list from this episode
                    for waypoint in self._all_episode_waypoint_lst[self._episode_counter-1]:
                        # Create the goal pose from the waypoint coordinates
                        goal = self.goal_pose(waypoint)
                        # Publish the goal to topic
                        self._pub.publish(goal)
                        # Sleep for a certain rate
                        self._rate.sleep()
                        # Likely throw an error and while loop starts again
                        current_time = rospy.get_time() 

                        # Continue to publish goal if not reached
                        while self._result.data == 0.0:
                            # Publish
                            self._pub.publish(goal)
                            # Wait
                            self._rate.sleep()
                            # Collect data at this step
                            self.collect_data()

                            # Stop episode if max time exceeded
                            if (rospy.get_time() > MAX_TIME_PER_EPISODE):
                                break;
                                
                        # Collect data when waypoint goal is reached
                        self.collect_data()

                    # Step episode counter
                    self._episode_counter = self._episode_counter + 1
                    if ((self._episode_counter) > TOTAL_EPISODES):
                        rospy.logerr("All simulations have been done. Pausing gazebo")
                        self._pause()
                        self._terminate = True
                    else:
                        rospy.logerr("preparing new simulation!")
                        #rospy.logerr("pause simulation!")
                        #self._pause()
                        #rospy.logerr("wait!")
                        #time.sleep(1)
                        #rospy.logerr("reset simulation!")
                        self._resetSimulation()
                        #self._resetSimulation()
                        # rospy.logerr("wait!") 
                        # time.sleep(20)
                        # rospy.logerr("start new simulation!")
                        self._unpause()
                        rospy.logerr("Continue simulation!") 

                if self._terminate:
                    break

            except rospy.ROSInterruptException:
                rospy.logerr("ROS InterruptException! Just ignore the exception! Hi") 
            except rospy.ROSTimeMovedBackwardsException:
                rospy.logerr("ROS Time Backwards! Just ignore the exception! Hello")


    def generate_random_waypoints(self):
        '''Generate a list of way points lists
        
        Example
            [
                [
                    [(69, 154, 0.0), (0.0, 0.0, 0.7, 0.7)],
                    [(124, 85, 0.0), (0.0, 0.0, 0.7, 0.7)],
                    [(70, 29, 0.0), (0.0, 0.0, 0.7, 0.7)],
                    [(46, 177, 0.0), (0.0, 0.0, 0.7, 0.7)]],
                ]
                ...
            ]
        '''

        all_episode_waypoints_lst = []
        for episode in range(TOTAL_EPISODES):
            #Reset episode waypoints list
            episode_waypoint_lst = []
            for waypoint_n in range(WAYPOINTS_PER_EPISODE):
                #Generate random x y coordinate for way point
                random_waypoints_xy = random.sample(range(WAYPOINT_MIN_XY_COORDINATES, WAYPOINT_MAX_XY_COORDINATES), 2)
                # Create the way point
                waypoint = [(random_waypoints_xy[0], random_waypoints_xy[1], 0.0), (0.0, 0.0, 0.7, 0.7)]
                # Add waypoint to the other way points in this episode
                episode_waypoint_lst.append(waypoint)
            
            # Add all waypoints of this episode to list of all episode waypoints
            all_episode_waypoints_lst.append(episode_waypoint_lst)


        return all_episode_waypoints_lst

    def goal_pose(self, pose):
        goal_pose = Odometry()
        goal_pose.header.stamp = rospy.Time.now()
        goal_pose.header.frame_id = 'world'
        goal_pose.pose.pose.position = Point(pose[0][0]+self._x_offset, pose[0][1]+self._y_offset, 0.)
        return goal_pose

    def get_result(self, result_aux):
        global result
        self._result.data = result_aux.data
    
    def initialize_data_lists(self):
        '''Initialize Data lists'''
        # Position
        self._position_x_lst = []
        self._position_y_lst = []
        self._position_z_lst = []
        # Orientation
        self._orientation_x_lst = []
        self._orientation_y_lst = []
        self._orientation_z_lst = []
        # Position rates
        self._position_x_rate_lst = []
        self._position_y_rate_lst = []
        self._position_z_rate_lst = []
        # Orientation Rates
        self._orientation_x_rate_lst = []
        self._orientation_y_rate_lst = []
        self._orientation_z_rate_lst = []
        # Environment
        self._wind_speed_lst = []
        # Engine
        self._engine_velocity_command_lst = []
        self._rudder_angle_lst = []
        # Waypoints
        self._waypoint_x_lst = []
        self._waypoint_y_lst = []
        self._reached_waypoint_lst = []

    def sub4data(self):
        # Subscribe to Odemetry topic
        self._odem = Odometry()
        rospy.Subscriber('state', Odometry, self.get_orientation)  

        # Subscribe to engine speed command
        self._engine_vel = JointState()
        rospy.Subscriber('velocity_command', JointState, self.get_engine_vel)

        # Subscribe to rudder angle
        self._rudder_ang = JointState()
        rospy.Subscriber('joint_setpoint', JointState, self.get_rudder_ang)

        # SUbscribe to way points
        self._waypoint_subscriber = Odometry()
        self._waypoint_subscriber = rospy.Subscriber('move_usv/goal', Odometry, self.get_waypoint)


    def get_orientation(self, odem_temp):
        '''Used for rospy Odometry state subscriber'''
        self._odem = odem_temp
    
    def get_engine_vel(self, engine_vel_temp):
        '''Used for rospy JointState velocity command subscriber'''
        self._engine_vel = engine_vel_temp

    def get_rudder_ang(self, rudder_ang_temp):
        '''Used for ropsy JointState rudder angle subscriber'''
        self._rudder_ang = rudder_ang_temp

    def get_waypoint(self, waypoint_temp):
        '''Used for rospy Odometry waypoint subscriber'''
        self._waypoint_subscriber = waypoint_temp
    
    def get_wind_speed(self):
        return WIND_SPEED_SET


    def collect_data(self):
        # Position
        self._position_x_lst.append(self._odem.pose.pose.position.x)
        self._position_y_lst.append(self._odem.pose.pose.position.y)
        self._position_z_lst.append(self._odem.pose.pose.position.z)
        # Orientation
        self._orientation_x_lst.append(self._odem.pose.pose.orientation.x)
        self._orientation_y_lst.append(self._odem.pose.pose.orientation.y)
        self._orientation_z_lst.append(self._odem.pose.pose.orientation.z)
        # Position Rates
        self._position_x_rate_lst.append(self._odem.twist.twist.linear.x)
        self._position_y_rate_lst.append(self._odem.twist.twist.linear.y)
        self._position_z_rate_lst.append(self._odem.twist.twist.linear.z)
        # Orientation Rates
        self._orientation_x_rate_lst.append(self._odem.twist.twist.angular.x)
        self._orientation_y_rate_lst.append(self._odem.twist.twist.angular.y)
        self._orientation_z_rate_lst.append(self._odem.twist.twist.angular.z)
        # Environment
        self._wind_speed_lst.append(self.get_wind_speed())
        # Engine
        self._engine_velocity_command_lst.append(self._engine_vel.velocity[0])
        self._rudder_angle_lst.append(self._rudder_ang.position[0])
        # Waypoints
        self._waypoint_x_lst.append(self._waypoint_subscriber.pose.pose.position.x)
        self._waypoint_y_lst.append(self._waypoint_subscriber.pose.pose.position.y)
        self._reached_waypoint_lst.append(self._result.data)
    
    def save_data_2_csv(self):
        '''Output recorded data to csv'''

       
        # Create data dictionary
        self.create_data_structures()

        with open(CURRENTDIR + CSV_FILE_NAME + '.csv','w') as file:
            
            writer = csv.writer(file)

            # Write feature names
            writer.writerow(self._features)

            # Write data (use zip to transpose columns to rows)
            writer.writerows(zip(*self._features_lsts))
        
        rospy.logerr('Data saved!')

    def create_data_structures(self):
        '''Merge all the data lists into a dataframe'''

        self._features = ['position_x','position_y','position_z',
                          'orientation_x', 'orientation_y', 'orientation_z',
                          'position_rate_x', 'position_rate_y', 'position_rate_z', 
                          'orientation_rate_x', 'orientation_rate_y', 'orientation_rate_z',
                          'wind_speed', 'engine_velocity_command', 'rudder_angle',
                          'waypoint_x', 'waypoint_y', 'reached_waypoint']
        
        self._features_lsts = [
            self._position_x_lst,
            self._position_y_lst,
            self._position_z_lst,
            self._orientation_x_lst,
            self._orientation_y_lst,
            self._orientation_z_lst,
            self._position_x_rate_lst,
            self._position_y_rate_lst,
            self._position_z_rate_lst,
            self._orientation_x_rate_lst,
            self._orientation_y_rate_lst,
            self._orientation_z_rate_lst,
            self._wind_speed_lst,
            self._engine_velocity_command_lst,
            self._rudder_angle_lst,
            self._waypoint_x_lst,
            self._waypoint_y_lst,
            self._reached_waypoint_lst
        ]

        self._collected_data_dict = {
            'position_x': self._position_x_lst,
            'position_y': self._position_y_lst,
            'position_z': self._position_z_lst,
            'orientation_x': self._orientation_x_lst,
            'orientation_y': self._orientation_y_lst,
            'orientation_z': self._orientation_z_lst,
            'position_rate_x': self._position_x_rate_lst,
            'position_rate_y': self._position_y_rate_lst,
            'position_rate_z': self._position_z_rate_lst,
            'orientation_rate_x': self._orientation_x_rate_lst,
            'orientation_rate_y': self._orientation_y_rate_lst,
            'orientation_rate_z': self._orientation_z_rate_lst,
            'wind_speed': self._wind_speed_lst,
            'engine_velocity_command': self._engine_velocity_command_lst,
            'rudder_angle': self._rudder_angle_lst,
            'waypoint_x': self._waypoint_x_lst,
            'waypoint_y': self._waypoint_y_lst,
            'reached_waypoint': self._reached_waypoint_lst
        }


    def set_seed(self):
        random.seed(SEED)
        

if __name__ == '__main__':
    rospy.logwarn('Data Collection Reached')
	
