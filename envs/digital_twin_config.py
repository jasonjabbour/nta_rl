import attr

@attr.s
class EnvironmentConfig(object):
    '''Parameter used for the Open AI Gym Environment'''
    
    include_features_as_action = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0,0,0)
    
    include_features_as_observation = (1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,0,0)
    
    data_dir = 'data_collection/data/'
    
    data_file_names = ['digitaltwin_data_300waypoints_1ep_1',
                    'digitaltwin_data_300waypoints_1ep_1_wind4', 
                    'digitaltwin_data_620waypoints_1ep_wind5_maxcoord150']
    
    test_data_file_names = ['digitaltwin_data_300waypoints_1ep_1']
    
    features = ['position_x','position_y','position_z',
                'orientation_x', 'orientation_y', 'orientation_z',
                'position_rate_x', 'position_rate_y', 'position_rate_z', 
                'orientation_rate_x', 'orientation_rate_y', 'orientation_rate_z',
                'wind_speed', 'engine_velocity_command', 'rudder_angle',
                'waypoint_x', 'waypoint_y', 'reached_waypoint','waypoint_counter']

    steps_per_episode = 100
    
    attack_detected_bonus = 10
    
    attack_TP_threshold = .8
    
    attack_FN_threshold = .5

    attack_FP_threshold = .5
    
    attack_TN_threshold = .2