# file to generate the monster matrix (mm) of var indices data for each lc object
# by @oscars47 and @ghirsch123

import os
import fnmatch
import shutil
from tqdm import tqdm
import pandas as pd
import numpy as np
# import files we've created
from lcgen import get_lc
from vargen_slimmed import Variable

# set directories, define helper functions for conversions

DATA_DIR = 'ENTER THE DIRECTORY THAT HAS ALL OF YOUR DATA IN IT'
LC_DIR = os.path.join(DATA_DIR, 'g_band_lcs')
LC_OUT = os.path.join(DATA_DIR, 'lc_output')
VAR_OUT = os.path.join(DATA_DIR, 'var_output3')
FOLD_DIR = os.path.join(DATA_DIR, 'folded_g_band_lcs') # for folded .dats
FOLD_OUT = os.path.join(LC_OUT, 'folded')
IRREG_DIR = os.path.join(DATA_DIR, 'irreg_g_band_lc')
SLIMMED = os.path.join(DATA_DIR, 'slimmed_g_band_lcs')

# confirm we have each of these directories
if not(os.path.isdir(DATA_DIR)):
    os.makedirs(DATA_DIR)
if not(os.path.isdir(LC_DIR)):
    os.makedirs(LC_DIR)
if not(os.path.isdir(LC_OUT)):
    os.makedirs(LC_OUT)
if not(os.path.isdir(VAR_OUT)):
    os.makedirs(VAR_OUT)
if not(os.path.isdir(FOLD_OUT)):
    os.makedirs(FOLD_OUT)
if not(os.path.isdir(FOLD_DIR)):
    os.makedirs(FOLD_DIR)
if not(os.path.isdir(IRREG_DIR)):
    os.makedirs(IRREG_DIR) 
# make list of all LC_DIR files
lc_files = os.listdir(LC_DIR)
if not(os.path.isdir(SLIMMED)):
    os.makedirs(SLIMMED)

def get_file(name):
    temp_list = []
    temp_file = ''
    temp_list = name.split(' ')
    temp_file += temp_list[0] +'_' + temp_list[1] +'.dat'
    return temp_file

def get_name(file):
    temp_list=[]
    temp_name = ''
    temp_list = file.split('.dat')
    temp_name = temp_list[0]
    temp_list = temp_name.split('_')
    temp_name = temp_list[0] +' ' + temp_list[1]
    return temp_name

# get unique classes of variable objects---------------
vars = pd.read_csv(os.path.join(DATA_DIR, 'df_total_500.csv')) # df_total_500.csv should be the name of the slimmed data set, if you changed it change it again here
var_unique_name = list(vars['ID'].unique())

# to get the names, first split by '.dat' and remove final element (.dat), recombine; then split based on '_' and replace with ' '
lc_names = []
lc_not_names = []
for lc in lc_files:
    temp_name = get_name(lc)
    if temp_name in var_unique_name:
        lc_names.append(temp_name)
        temp_file = get_file(temp_name)
        file_name = os.path.join(LC_DIR, temp_file)
        shutil.copy(file_name, SLIMMED)
    else:
        lc_not_names.append(temp_name)

lc_slimmed = os.listdir(SLIMMED)

# fold lcs!-----------------------
# to be run once
def fold_lcs():
    for i in tqdm(range(len(lc_slimmed)), desc='progress...', position=0, leave=True):
        name = lc_names[i]
        file = lc_slimmed[i]
        # extract periods
        period = vars.loc[vars['ID']==name]['Period'].to_list()[0]
        
        # read in df
        c_path = os.path.join(LC_DIR, file)
        lc_df = pd.read_csv(c_path, sep='\t')

        # we need to extract the time list and mod each by the period
        time = np.array(lc_df['HJD'].to_list())
        #print(period, type(period))
        
        if (period > 0):
            time = time % period
            # now set the time column of df equal to this
            lc_df['HJD'] = time

            #now save lc_df as .dat in FOLD_DIR
            lc_df.to_csv(os.path.join(FOLD_DIR, file), sep='\t')
        else:
            lc_df.to_csv(os.path.join(IRREG_DIR, file), sep='\t')

#fold_lcs() # call to generate new .dats
    
# get folded lcs and compare them to lcs that are not folded
def get_compare_lcs(name):
    file = get_file(name)
    target = vars.loc[vars['ID']==name]['ML_classification'].to_list()[0]
    get_lc(file, name, target, FOLD_DIR, FOLD_OUT)
    get_lc(file, name, target, LC_DIR, LC_OUT)

# for actual data processing: go through each file in vs not periodic----------------


# create new df --- the monster matrix --- to hold results of Variable obj
mm_df_per = pd.DataFrame({
         'id': [], 'target': [], 'periodic': [],'mad': [], 'weighted mean': [],
            'chi2red': [], 'weighted stdev': [], 
            'iqr': [], 'roms': [], 'norm excess var': [], 
            'peak peak var': [], 'eta ratio': [], 'SB': [],'clipped stdev': [], 'period': [],
            'power': [], 'skew': [], 'kurtosis': [] 
        })

mm_df_irreg = pd.DataFrame({
         'id': [], 'target': [], 'periodic': [],'mad': [], 'weighted mean': [],
            'chi2red': [], 'weighted stdev': [], 
            'iqr': [], 'roms': [], 'norm excess var': [], 
            'peak peak var': [], 'eta ratio': [], 
            'SB': [],'clipped stdev': [], 'period': [],
            'power': [], 'skew': [], 'kurtosis': []
        })

# function to take in id and cleanup name to get lc--------------------
def get_var_data_name(df, name, isper, input):
    temp_file = get_file(name)
    target = vars.loc[vars['ID']==name]['ML_classification'].to_list()[0]
    # create new instance of Variable object
    temp_var = Variable(temp_file, name, isper, target, input)
    temp_df = pd.DataFrame(temp_var.return_dict())
    df_new = pd.concat([df, temp_df], ignore_index=True)
    #df_new = df.append(temp_var.return_dict(), ignore_index=True)
    return df_new

# takes in file name, not id name
def get_var_data_file(df, file, isper, input):
    temp_name = get_name(file)
    target = vars.loc[vars['ID']==temp_name]['ML_classification'].to_list()[0]
    # create new instance of Variable object
    temp_var = Variable(file, temp_name, isper, target, input)
    temp_df = pd.DataFrame(temp_var.return_dict())
    df_new = pd.concat([df, temp_df], ignore_index=True)
    #df_new = df.append(temp_var.return_dict(), ignore_index=True)
    return df_new

# for complete dataset
def run_per(mm_df):
    # first do periodic
    isper = 1
    per_ls = os.listdir(FOLD_DIR)
    for i in tqdm(range(len(per_ls)), desc='progress on periodic...', position=0, leave=True):
        file = per_ls[i]
        mm_df = get_var_data_file(mm_df, file, isper, FOLD_DIR)

    return mm_df

def run_irreg(mm_df):
       # then do non-periodic
    isper = 0
    irreg_ls = os.listdir(IRREG_DIR)
    for i in tqdm(range(len(irreg_ls)), desc='progress on non-periodic...', position=0, leave=True):
        file = irreg_ls[i]
        mm_df = get_var_data_file(mm_df, file, isper, IRREG_DIR)

    return mm_df

# run these lines to call the data generation functs    
mm_df_per = run_per(mm_df_per)
mm_df_irreg = run_irreg(mm_df_irreg)

# save un-normalized results
def save_un():
    print('saving results!')
    mm_per_name='folded_mm_per.csv'
    mm_irreg_name = 'folded_mm_irreg.csv'
    mm_df_per.to_csv(os.path.join(VAR_OUT, mm_per_name))
    mm_df_irreg.to_csv(os.path.join(VAR_OUT, mm_irreg_name))

# save un-normalized results
save_un()

# take in a column and returns normalized version
def normalize_col(col):
    # compute min
    col_min = min(col)
    col_max = max(col)

    if col_max - col_min > 0:
        col_n = []
        for x in col:
            x_n = (x - col_min) / (col_max - col_min)
            col_n.append(x_n)
        return col_n

    elif (col_max - col_min == 0) and (col_min > 0):
        col_n = []
        for x in col:
            x_n = (x - col_min) / (col_max)
            col_n.append(x_n)
        return col_n
        
    # everything is 0
    else:
        return col

    
# normalize the data for our NN!
def normalize_master(data_path):
    # read the csv: un-normalized df
    un_df = pd.read_csv(data_path)
    #un_df = un_df.head(10)
    var_indices_ls = list(un_df.columns)[4:]
    # go through each column and separately normalize
    for col_name, values in un_df[var_indices_ls].iteritems():
        #print(values)
        temp_col = normalize_col(values)
        un_df[col_name] = temp_col
    # drop one of the duplicate indexing cols
    un_df = un_df.iloc[:, 1:]
    return un_df
    
per_path = os.path.join(VAR_OUT, 'folded_mm_per.csv')
n_df = normalize_master(per_path)
n_df_name = 'folded_mm_per_norm.csv'
n_df.to_csv(os.path.join(VAR_OUT, n_df_name))
