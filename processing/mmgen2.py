# main file to generate the monster matrix (mm) of var indices data for each lc object
# by @oscars47 and @ghirsch123

import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from prettytable import PrettyTable
# import files we've created
from vargen2 import Variable2

# set directories, define helper functions for conversions
DATA_DIR = '/Volumes/PHLUID/astro/data/g_band'
LC_DIR = os.path.join(DATA_DIR, 'g_band_lcs')
LC_OUT = os.path.join(DATA_DIR, 'lc_output')
VAR_OUT = os.path.join(DATA_DIR, 'var_output')

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

# helper files to amend datasets----------------------------
def get_ascii(df, save_path, name='test'):
    '''export ra and dec to make ascii table given pd dataframe.
        use https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd to extract color info. takes ~ 20 mins.
        when you use output from function below, need to delete top and bottom +----+ decorations, then cut the |ra dec| |float float| lines, cntrl-f '|'| replace ' ', then paste back and submit to online portal'''
    # get list of ra, dec and add to x
    ra_ls = df['RAJ2000'].to_list()
    dec_ls = df['DEJ2000'].to_list()
   
    x = PrettyTable()

    column_names = ["ra", "dec"]
    ra_ls.insert(0, 'float')
    dec_ls.insert(0, 'float')
    
    x.add_column(column_names[0], ra_ls)
    x.add_column(column_names[1], dec_ls)  

    data = x.get_string()
    print(data)
    print('saving!')
    with open(os.path.join(save_path, name+'.txt'), 'w') as f:
        f.write(data)
        
def add_rounding_asassn(df):
    '''add rounded ra, dec column to asassn'''
    ra = df['RAJ2000'].to_list()
    dec = df['DEJ2000'].to_list()
    r_ra = np.round(ra, 5)
    r_dec = np.round(dec, 5)
    df['rounded ra'] = r_ra
    df['rounded dec'] = r_dec
    df.to_csv(os.path.join(DATA_DIR, 'asassn_rounded.csv'))


def add_rounding_table(df):
    '''add rounding to color table output'''
    ra = df['ra_01'].to_list()
    dec = df['dec_01'].to_list()
    r_ra = np.round(ra, 5)
    r_dec = np.round(dec, 5)
    df['rounded ra'] = r_ra
    df['rounded dec'] = r_dec
    df.to_csv(os.path.join(DATA_DIR, 'table_rounded.csv'))

def init_var2_dfs():
    '''initiate var dfs for Variable2 object'''
    mm_df = pd.DataFrame({
        'name': [], 'target': [], 'period': [], 'power': [],
        'T_t': [], 'T_p': [], 'T_2p': [], 
        'delta_t': [], 'delta_p': [],
        'j-k': [], 'h-k': [], 'skew': [],
        'kurtosis': [], 'stdev': [], 'median': [],
        'iqr': [], 'mad': [], 'von neumann': [],
        'a_hl': [],
        'weighted mean': [],
        'chi2red': [], 'weighted stdev': [],
        'roms': [], 'norm excess var': [], 
        'peak peak var': [], 'lag1 auto': [],
        'I': [],'Ifi': [], 'J': [], 'K': [], 'L': [],
        'J time': [], 'I clipped': [],
        'J clipped': [],
        'Ex': [], 'SB': [],'clipped stdev': []
        })
    return mm_df

# -------------------- #
def get_var_data_name(df, name, input, table_df):
    '''function to take in id and cleanup name to get lc'''
    temp_file = get_file(name)
    target = vars.loc[vars['ID']==name]['ML_classification'].to_list()[0]
    # create new instance of Variable object
    #temp_var = Variable(temp_file, name, isper, target, input)
    temp_var = Variable2(vars, table_df, temp_file, name, target, input)
    temp_df = pd.DataFrame(temp_var.return_dict())
    #df_new = pd.concat([df, temp_df], ignore_index=True)
    df_new = df.append(temp_df.return_dict(), ignore_index=True)
    return df_new

def get_var_data_file(df, file, input, asassn, table):
    '''takes in file name, not id name'''
    temp_name = get_name(file)
    target = vars.loc[vars['ID']==temp_name]['ML_classification'].to_list()[0]
    # create new instance of Variable object
    temp_var = Variable2(asassn, table, file, temp_name, target, input)
    temp_df = pd.DataFrame(temp_var.return_dict())
    df_new = pd.concat([df, temp_df], ignore_index=True)
    return df_new

def normalize_col(col):
    '''take in a column and returns normalized version'''
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

def normalize_master_file(data_path):
    '''normalize the data for our NN!'''
    # read the csv: un-normalized df
    un_df = pd.read_csv(data_path)
    var_indices_ls = list(un_df.columns)[4:]
    # go through each column and separately normalize
    for col_name, values in un_df[var_indices_ls].iteritems():
        #print(values)
        temp_col = normalize_col(values)
        un_df[col_name] = temp_col
    # drop one of the duplicate indexing cols
    un_df = un_df.iloc[:, 1:]
    return un_df

def normalize_master_df(un_df):
    '''normalize the data for our NN!, takes in df'''
    var_indices_ls = list(un_df.columns)[2:]
    # go through each column and separately normalize
    for col_name, values in un_df[var_indices_ls].iteritems():
        #print(values)
        temp_col = normalize_col(values)
        un_df[col_name] = temp_col
    # drop one of the duplicate indexing cols
    un_df = un_df.iloc[:, 1:]
    return un_df

# main functions ------------------------------------------------------ #
def get_mm(names, asassn, pro_table):
    '''Function to get the master dataframe for the Variable2 object. This includes all the variability indices and the color indices.
    Inputs:
        names: list of names for the output csv files (un-normalized and normalized)
        asassn: pandas dataframe of asassn variables
        pro_table: pandas dataframe of color data
    '''
    mm_df = init_var2_dfs()
    for i in tqdm(range(len(lc_files)), desc='progress...', position=0, leave=True):
        file = lc_files[i]
        mm_df = get_var_data_file(mm_df, file, LC_DIR, asassn, pro_table) 

    # save unnormalized
    mm_df.to_csv(os.path.join(VAR_OUT, names[0]))

    # normalize
    mm_df = normalize_master_df(mm_df)
    # save
    mm_df.to_csv(os.path.join(VAR_OUT, names[1]))
        
    return mm_df

def do_prelim():
    '''Function to handle the preliminary stuff for the Variable2 object. This includes: getting the ascii table for color data, getting the asassn variables csv, and rounding the dataframes.'''
    
    # get unique classes of variable objects---------------
    vars = pd.read_csv(os.path.join(DATA_DIR, 'asassn_variables_x.csv'))
    # round df
    if 'asassn_rounded.csv' not in os.listdir(DATA_DIR):
        add_rounding_asassn(vars)

    # get the ascii table for color data------------------
    name='table_irsa_catalog_search_input'
    if os.path.join(DATA_DIR, name+'.txt') not in os.listdir(DATA_DIR):
        get_ascii(vars, DATA_DIR, name)

    assert 'table_irsa_catalog_search_results.csv' in os.listdir(DATA_DIR), 'You must visit https://irsa.ipac.caltech.edu/cgi-bin/Gator/nph-dd and upload the ascii search inputs file (after removing the decorations and bars but preserving the ra and dec with float labels), download results and  convert to a csv file, upload it to the data directory.'
    table_df = pd.read_csv(os.path.join(DATA_DIR, 'table_irsa_catalog_search_results.csv'))
    add_rounding_table(table_df)

# ------------------------------------------------------------------- #
if __name__ == '__main__':
    # confirm we have each of these directories
    if not(os.path.isdir(DATA_DIR)):
        os.makedirs(DATA_DIR)
    if not(os.path.isdir(LC_DIR)):
        os.makedirs(LC_DIR)
    if not(os.path.isdir(LC_OUT)):
        os.makedirs(LC_OUT)
    if not(os.path.isdir(VAR_OUT)):
        os.makedirs(VAR_OUT)

    # make list of all LC_DIR files
    lc_files = os.listdir(LC_DIR)

    # to get the names, first split by '.dat' and remove final element (.dat), recombine; then split based on '_' and replace with ' '
    lc_names = []
    for lc in lc_files:
        temp_name = get_name(lc)
        lc_names.append(temp_name)

    # prelimary stuff-----------------------------------
    names = ['mm_2_un.csv', 'mm_2_n.csv']
    if 'asassn_rounded' not in os.listdir(DATA_DIR) or 'table_rounded' not in os.listdir(DATA_DIR):
        do_prelim()

    # create monster matrix-----------------------------
    mm_df = get_mm()
    # read in and normalize
    un_df = pd.read_csv(os.path.join(VAR_OUT, names[0]))
    n_df = normalize_master_df(un_df)
    n_df.to_csv(os.path.join(VAR_OUT, names[1]))