"""LSTM Model for NTA Data"""

from cProfile import label
import os
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(os.path.dirname(currentdir))
os.sys.path.insert(0, parentdir)

import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Sequential

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

OUTPUT_DIRECTORY = 'output/'

# Parameters
FEATURES_CONSIDERED = ['Latitude (degrees)', 'Longitude (degrees)', 'Velocity North (kn)', 
                        'Velocity East (kn)', 'Course (degrees)', 'Speed Through The Water (kn)', 
                        'Heading (degrees)', 'Heading Rate (degrees/s)', 'Apparent Wind Speed (kn)']

INDEX_COLUMN_NAME = 'Time of Data (s)'

INCLUDED_FEATURES_NUM = 2
                       
BATCH_SIZE = 128
BUFFER_SIZE = 10000
TRAIN_SPLIT = 300

# LSTM Parameters
EVALUATION_INTERVAL = 200
EPOCHS = 4
PATIENCE = 5

STEP = 6
# Reproducibility
SEED = 1113
tf.random.set_seed(SEED)

# Plot Global Variables
mpl.rcParams['figure.figsize'] = (17, 5)
mpl.rcParams['axes.grid'] = False
sns.set_style("whitegrid")

class NTALSTM(object):
    """Class to train an LSTM model"""

    def __init__(self, 
                 data_df: pd.DataFrame, 
                 dev: bool = False):
        """A basic constructor of an LSTM model for NTA data
        
        Args:
            data_df: A pandas dataframe of NTA data
            dev: boolean indicating if in development mode. 
                While in development mode, no new directories will
                be created. See create_model_name()
        
        
        """

        self._data_df = data_df
        self._dev = dev

        # Create model directory and model name
        self.setup_model()    

        # Prepare data
        self.setup_df()

        # View Data
        self.view_data()
    
    def setup_model(self):
        """Ceate the directory to store model and the model name. 

        This function assumes that all directories within the output directory 
        follow the naming convension model_number where number is a number greater than 0
        and only one underscore exists.

        Example:
            model_1
            model_2
        """

        model_number = 1
        default_model_directory_name = 'model_'

        # Get list of directories within the output directory
        output_directories = sorted(os.listdir(OUTPUT_DIRECTORY))

        # Determine number of new model
        if len(output_directories) > 0:

            # Find the number associated with the newest directory
            largest_model_number_directory = output_directories[-1].split('_') 

            # Make sure directory follows model_number format
            if len(largest_model_number_directory) > 1:
                # Get the number assuming number is stored in last position 
                if largest_model_number_directory[-1].isdigit():
                    # Set model number to one greater of previous number
                    model_number = int(largest_model_number_directory[-1]) + 1     


        # Create model name
        self._model_name = default_model_directory_name + str(model_number)
      
        # Create model directory name
        self._model_directory = OUTPUT_DIRECTORY+self._model_name
        
        # Only create model directory if not in development mode
        if not self._dev:
            os.mkdir(self._model_directory)
        
    def setup_df(self):
        """Select features from dataframe and set the dataframe index"""

        self._features = self._data_df[FEATURES_CONSIDERED]
        self._features.index = self._data_df[INDEX_COLUMN_NAME]
        self._data_df = self._features.values


        if self._dev:
            print(self._features.head())
    
    def multioutput_model(self, 
                          data_df,
                          target,
                          start_index, 
                          end_index=None, 
                          history_size=None, 
                          target_size=None, 
                          step=None, 
                          single_step=False):

        data = []
        labels = []

        # Must have some history at start so perturb starting index
        start_index = start_index + history_size

        # Calculate end index
        if end_index is None:
            end_index = len(data_df) - target_size
        
        for i in range(start_index, end_index):
            indices = range(i-history_size, i, step)
            data.append(data_df[indices])
        
            if single_step:
                labels.append(target[i+target_size])
            else:
                labels.append(target[i:i+target_size])
            
            
        return np.array(data)[:,:,:,np.newaxis, np.newaxis], np.array(labels)[:,:,:, np.newaxis, np.newaxis]

    def multi_step_output_plot(self, history, true_future, prediction):

        plt.figure(figsize=(18,6))

        num_in = self.create_time_steps(len(history))
        num_out = len(true_future)

        for i, (var, c) in enumerate(zip(self._features.columns[:INCLUDED_FEATURES_NUM], ['b','r'])):
            plt.plot(num_in, np.array(history[:,i]), c, label=var)
            plt.plot(np.arange(num_out)/STEP, np.array(true_future[:,i]), c+'o', markersize=5, alpha=0.5, 
                     label=f'True {var.title()}')
            if prediction.any():
                plt.plot(np.arange()/STEP, np.array(prediction[:i]), '*', markersize=5, alpha=0.5,
                         label=f'Predicted {var.title()}')

        plt.legend(loc='upper left')

        plt.show()
    
    def view_data(self):

        past_history = 70
        future_target = 30


        # Get test and train data
        self.x_train_multi, self.y_train_multi = self.multioutput_model(self._data_df[:, :INCLUDED_FEATURES_NUM], self._data_df[:,:INCLUDED_FEATURES_NUM], 0, 
                                                                TRAIN_SPLIT, past_history, 
                                                                future_target, STEP)
        self.x_val_multi, self.y_val_multi = self.multioutput_model(self._data_df[:,:INCLUDED_FEATURES_NUM], self._data_df[:,:INCLUDED_FEATURES_NUM],
                                                          TRAIN_SPLIT, None, past_history, future_target, STEP)

        
        self.train_data_multi = tf.data.Dataset.from_tensor_slices((self.x_train_multi, self.y_train_multi))
        self.train_data_multi = self.train_data_multi.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()

        val_data_multi = tf.data.Dataset.from_tensor_slices((self.x_val_multi, self.y_val_multi))
        val_data_multi = val_data_multi.batch(BATCH_SIZE).repeat()

        for x,y in val_data_multi.take(1):
            self.multi_step_output_plot(np.squeeze(x[0]), np.squeeze(y[0]), np.array([0]))    
    
    
        print (self.x_train_multi.shape,
            self.y_train_multi.shape,
            self.x_val_multi.shape,
            self.y_val_multi.shape,
            'Single window of past history : {}'.format(self.x_train_multi[0].shape),
            'Target temperature to predict : {}'.format(self.y_train_multi[0].shape),
            sep='\n')

    def create_time_steps(self, length):
        return list(range(-length,0))

    def get_model_name(self):
        return self._model_name
    
    def get_model_directory(self):
        return self._model_directory
    

            
            
        