import tensorflow as tf
import pandas as pd
import numpy as np
import seaborn as sns
import sys
import argparse

from learning import nta_lstm


print(f'Tensorflow Version: {tf.__version__}')
print(f'Pandas Version: {pd.__version__}')
print(f'Numpy Version: {np.__version__}')
print(f'System Version: {sys.version}')


def read_data(data_file_name: str) -> pd.DataFrame:
    """Read csv data and return the pandas dataframe object
    
    Args:
        file_name: the name of a csv file as a string

    Raises:
        AssertionError: If file_name is not a csv file

    Returns:
        dataframe object
    """
    
    assert '.csv' in data_file_name, 'Data file name name must be a .csv file.'

    data_directory = 'data/'

    df = pd.read_csv(data_directory + data_file_name)

    return df


if __name__ == '__main__':
    # Input Commands
    arg_parser = argparse.ArgumentParser()
    arg_parser.add_argument('--data_file_name', dest='data_file_name', type=str, default='od19.csv')
    arg_parser.add_argument('--dev', dest='dev', action='store_true', default=False)

    args = arg_parser.parse_args()

    # .csv file name with nta data
    data_file_name = args.data_file_name
    # Development mode will not create new directories
    dev = args.dev

    # Read CSV Data
    nta_df = read_data(data_file_name)

    # Initialize LSTM model
    nta_lstm_model = nta_lstm.NTALSTM(nta_df,
                                      dev=dev)


