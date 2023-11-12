# file to take the output of the mmgen2 and convert to numpy arrays for training/validating/testing
# adapted from fall2022/nn_prep.py

import os
import numpy as np
import pandas as pd

## define list of unique targets and dictionaries to convert back and forth from class to int representation ##
unique_targets = ['RRC', 'EA', 'DCEP', 'DCEPS', 'YSO', 'CWA', 'EB', 'RRD', 'RRAB', 'HADS', 'M', 'ROT', 'VAR', 'L', 'EW', 'CWB', 'SR', 'RVA', 'DSCT']

class_to_int = dict((target, i) for i, target in enumerate(unique_targets))
int_to_class = dict((i, target) for i, target in enumerate(unique_targets))

# helper function # 
def make_1_hots(targets):
    '''Converts a list of target as strings into a list of 1 hot encoded vectors.
    
    Params:
        targets (list): list of targets as strings
    Returns:
        target_1_hots (list): list of 1 hot encoded vectors
    
    '''    
    target_1_hots = []
    for target in targets:
        # get the index
        index = class_to_int[target]
        # initialize 0 vector
        vec = np.zeros(len(unique_targets))
        # place 1 at index location
        vec[index] = 1
        target_1_hots.append(vec)
    return target_1_hots

def prep_data(mm_n, data_dir, test_split=0.5, tv_split=0.8):
    '''Prepares the data for training and validating the neural network, which must be in (.npy) arrays
    
    Params:
        mm_n (DataFrame): DataFrame of the mmgen2 output
        dta_dir (str): path to the data directory
        test_split (float): fraction of data to be used for testing (THIS IS SACRED; DO NOT USE UNTIL YOU HAVE A SATISFACTORY MODEL TO TEST ON!!)
        tv_split (float): fraction of data to be used for training vs validating (of the portion left over from the test_split)

    Returns:
        None, but saves the following files to the data directory:
            train_x_ds.npy: numpy array of the input training data
            val_x_ds.npy: numpy array of the input validation data
            train_y_ds.npy: numpy array of the ouput training targets
            val_y_ds.npy: numpy array of the ouput validation targets
            x_test.npy: numpy array of the input testing data
            y_test.npy: numpy array of the testing targets
    
    '''
    # get names
    names = mm_n['name'].to_list()

    # get list of targets
    targets = mm_n['target'].to_list()

    # drop the target column of mm_n as well as the first two columns since those are index and name
    mm_n = mm_n.drop(columns=['target', 'name'])
    
    # first split for testing; return a dataset to be used to validate
    init_split_index = int(test_split*len(mm_n))
    # train-validate portion
    mm_n_tv = mm_n.iloc[:init_split_index, :]
    mm_n_test = mm_n.iloc[init_split_index:, :]
    tv_names = names[:init_split_index]
    test_names = names[init_split_index:]

    # make our 1 hot encoded vectors
    target_1_hots = make_1_hots(targets)
    targets_tv = target_1_hots[:init_split_index]
    targets_test= target_1_hots[init_split_index:]

    # split within tv dataset
    train_split_index = int(tv_split*len(mm_n_tv))
    train_x_ds = mm_n_tv.iloc[:train_split_index, :] 
    train_y_ds = targets_tv[:train_split_index]
    train_names = tv_names[:train_split_index]
    val_x_ds = mm_n_tv.iloc[train_split_index:, :]
    val_y_ds = targets_tv[train_split_index:]
    val_names = tv_names[train_split_index:]

    # convert to numpy arrays
    train_x_ds = train_x_ds.to_numpy(dtype=np.float32)
    val_x_ds = val_x_ds.to_numpy(dtype=np.float32)
    train_y_ds = np.array(train_y_ds)
    val_y_ds = np.array(val_y_ds)
    mm_n_test= mm_n_test.to_numpy(dtype=np.float32)
    targets_test= np.array(targets_test)

    # save results #
    print('saving datasets... :)')
    np.save(os.path.join(data_dir, 'train_x_ds.npy'), train_x_ds)
    np.save(os.path.join(data_dir, 'val_x_ds.npy'), val_x_ds)
    np.save(os.path.join(data_dir, 'train_y_ds.npy'), train_y_ds)
    np.save(os.path.join(data_dir, 'val_y_ds.npy'), val_y_ds)
    # save name as csv
    pd.DataFrame(train_names).to_csv(os.path.join(data_dir, 'train_names.csv'))
    pd.DataFrame(val_names).to_csv(os.path.join(data_dir, 'val_names.csv'))

    np.save(os.path.join(data_dir, 'x_test.npy'), mm_n_test)
    np.save(os.path.join(data_dir, 'y_test.npy'), targets_test)
    # save name as csv
    pd.DataFrame(test_names).to_csv(os.path.join(data_dir, 'test_names.csv'))



if __name__ == '__main__':
    # define path to where your data is
    DATA_DIR = '/Users/oscarscholin/Desktop/Pomona/Senior_Year/Fall2024/Astro_proj/UCVS/data'

    # load mmgen2 output
    # NOTE: the _vars1 file is redundant: the _targ includes the indices + ground truth labels, which is want we need to build our datasets for the neural net
    mm_n = pd.read_csv(os.path.join(DATA_DIR, 'mm_2_n_targ.csv'))

    # need to drop bad columns
    # log10fap was causing loss to blow up
    mm_n = mm_n.drop(columns=['Unnamed: 0', 'log_10 fap'])

    # prep the data
    # can change the test_split and tv_split if you want to by passing those arguments
    prep_data(mm_n, DATA_DIR)