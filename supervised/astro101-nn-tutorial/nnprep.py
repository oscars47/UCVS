# created by @oscars47 and @ghirsch123 created fall 2022
# file to generate confusion matrix to investigate what var indices to use

import pandas as pd
import os
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt

DATA_DIR = 'ENTER THE DIRECTORY THAT HAS ALL OF YOUR DATA IN IT'

# read in csv
global mm_n, unique_targets, targets
mm_n = pd.read_csv(os.path.join(DATA_DIR,'folded_mm_per_norm.csv'))
targets = mm_n['target'].to_list()
#unique_targets = list(set(targets))
unique_targets = ['SR', 'ROT', 'EW', 'EA', 'L']
mm_n = mm_n.iloc[:, 4:]

# these hold the conversion from the list of var classes to indices in the list and vice versa
def get_target_dicts():
    global class_to_int, int_to_class
    #print(unique_targets)
    class_to_int = dict((target, i) for i, target in enumerate(unique_targets))
    int_to_class = dict((i, target) for i, target in enumerate(unique_targets))

# initialize dicts
get_target_dicts()

# these one hot vectors is what the NN outputs, initializing them helps with training as it can verify whether it predicted correctly
def make_1_hots(targets):
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

def prep_data():
    # convert the targets into 1 hot encoded vectors
    
    # first split 50-50; return a dataset to be used to validate
    init_split_index = int(0.5*len(mm_n))
    # train-validate portion
    mm_n_tv = mm_n.iloc[:init_split_index, :]
    mm_n_extra = mm_n.iloc[init_split_index:, :]

    # make our 1 hot encoded vectors
    target_1_hots = make_1_hots(targets)

    targets_tv = target_1_hots[:init_split_index]
    targets_extra = target_1_hots[init_split_index:]


    # split within tv dataset
    train_split_index = int(0.8*len(mm_n_tv))
    train_x_ds = mm_n_tv.iloc[:train_split_index, :]
    val_x_ds = mm_n_tv.iloc[train_split_index:, :]
    train_y_ds = targets_tv[:train_split_index]
    val_y_ds = targets_tv[train_split_index:]

    # convert to numpy arrays
    train_x_ds = train_x_ds.to_numpy()
    val_x_ds = val_x_ds.to_numpy()
    train_y_ds = np.array(train_y_ds)
    val_y_ds = np.array(val_y_ds)

    
    mm_n_extra = mm_n_extra.to_numpy()
    targets_extra = np.array(targets_extra)

    # save!!
    print('saving datasets!')
    np.save(os.path.join(DATA_DIR, 'train_x_ds.npy'), train_x_ds)
    np.save(os.path.join(DATA_DIR, 'val_x_ds.npy'), val_x_ds)
    np.save(os.path.join(DATA_DIR, 'train_y_ds.npy'), train_y_ds)
    np.save(os.path.join(DATA_DIR, 'val_y_ds.npy'), val_y_ds)

    np.save(os.path.join(DATA_DIR, 'mm_n_extra.npy'), mm_n_extra)
    np.save(os.path.join(DATA_DIR, 'targets_extra.npy'), targets_extra)

    

prep_data()



