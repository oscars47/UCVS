# file to generate confusion matrix to investigate what var indices to use

import pandas as pd
import os
import numpy as np
from sklearn import metrics
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

from sklearn.decomposition import PCA

MAIN_DIR = '/home/oscar47/Desktop/astro101/data/g_band'
DATA_DIR = '/home/oscar47/Desktop/astro101/data/g_band/var_output/v0.1.1'

# read in csv
global mm_n, unique_targets, targets
mm_n = pd.read_csv(os.path.join(DATA_DIR,'mm_2_n _var1.csv')) # var1 file doesn't have log10 fap parameter which caused issue
#mm_n = pd.read_csv(os.path.join(DATA_DIR,'mm_2_n.csv')) # var1 file doesn't have log10 fap parameter which caused issue
# remove first two columns
mm_n = mm_n.iloc[:, 2:]
#mm_n = mm_n.iloc[:, 6:]
print(mm_n)
mm_targ = pd.read_csv(os.path.join(DATA_DIR,'mm_2_n_targ.csv'))
targets = mm_targ['target'].to_list()
asassn = pd.read_csv(os.path.join(MAIN_DIR, 'asassn_rounded.csv'))
# now we ned to find targets
def get_targets(df):
    targets = []
    for i in tqdm(range(len(mm_n))):
        name = df.loc[i, 'name']
        asassn_row = asassn.loc[asassn['ID']==name]
        target = asassn_row['ML_classification'].to_list()[0]
        targets.append(target)
    # append new column
    df['target']=targets
    # now save 
    df.to_csv(os.path.join(DATA_DIR, 'mm_2_n_targ.csv'))

#get_targets(mm_n)
    
#unique_targets = list(set(targets))
unique_targets = ['RRC', 'EA', 'DCEP', 'DCEPS', 'YSO', 'CWA', 'EB', 'RRD', 'RRAB', 'HADS', 'M', 'ROT', 'VAR', 'L', 'EW', 'CWB', 'SR', 'RVA', 'DSCT']

def get_heatmap():
    # print(mm_n.head(5))
    matrix = mm_n.corr().round(3)
    sns.heatmap(matrix, annot=True)
    plt.savefig(os.path.join(DATA_DIR, 'heatmap.jpeg'))
    plt.show()

# these hold the conversion from the list of var classes to indices in the list and vice versa
def get_target_dicts():
    global class_to_int, int_to_class
    #print(unique_targets)
    class_to_int = dict((target, i) for i, target in enumerate(unique_targets))
    int_to_class = dict((i, target) for i, target in enumerate(unique_targets))

# initialize dicts
get_target_dicts()

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

def prep_data(mm_n):
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

    

prep_data(mm_n)


# testing-------
# period = mm_n['period']
# power = mm_n['power']
# fap = mm_n['log_10 fap']


# mm_n_var1 = pd.DataFrame()
# mm_n_var1['period'] = period

# mm_n_var2 = pd.DataFrame()
# mm_n_var2['period'] = period
# mm_n_var2['power'] = power

# mm_n_var3 = pd.DataFrame()
# mm_n_var3['period'] = period
# mm_n_var3['power'] = power
# mm_n_var3['log10 fap'] = fap
# print(fap)

#prep_data(mm_n_var3)
#get_heatmap()

# pca = PCA(n_components=35)
# pca.fit(mm_n)
# print(sum(pca.explained_variance_))



