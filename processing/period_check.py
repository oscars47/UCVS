import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from astropy.timeseries import LombScargle

# set directories, define helper functions for conversions
DATA_DIR = '/Volumes/PHLUID/astro/data/g_band'
VAR_OUT = os.path.join(DATA_DIR, 'var_output')

def get_LS_period(time_ls, mag_ls, mag_err_ls, name):
    try:
        ls = LombScargle(time_ls, mag_ls,mag_err_ls, normalization='standard')
        freq, power = ls.autopower(method='fastchi2')
        fap = ls.false_alarm_probability(power.max())

        best_frequency = freq[np.argmax(power)]
        period = 1/ best_frequency
        
        return period, power.max(), fap
    except:
        print("object %s's period could not be calculated" %name)
        return 0,0,0

def get_time_mag_magerr(file):
    '''This function reads in a lc file and returns the time, mag, and mag_err lists'''
    try:
        # create dataframe
        c_path = os.path.join(input, file)
        lc_df = pd.read_csv(c_path, sep='\t')
        # initialize time, mag, and mag_err lists
        time_ls = np.array(lc_df['HJD'].to_list()).astype(float)
        # unclean mag_ls
        mag_uc = lc_df['mag'].to_list()
        mag_c = []
        for mag in mag_uc:
            if not ('>' in str(mag)) or ('<' in str(mag)):
                mag_c.append(mag)
            else:
                mag_temp = str(mag)[1:]
                mag_c.append(float(mag_temp))
        #self.mag_ls = np.array(lc_df['mag'].to_list()).astype(float)
        mag_ls = np.array(mag_c).astype(float)
        mag_err_ls = np.array(lc_df['mag_err'].to_list()).astype(float)
        return time_ls, mag_ls, mag_err_ls

    except:
        print('Object %s could not be read in' %file)
        return 0,0,0

def get_name(file):
    temp_list=[]
    temp_name = ''
    temp_list = file.split('.dat')
    temp_name = temp_list[0]
    temp_list = temp_name.split('_')
    temp_name = temp_list[0] +' ' + temp_list[1]
    return temp_name

    
if __name__=='__main__':
    # read in the ground truth information about the objects
    vars = pd.read_csv(os.path.join(DATA_DIR, 'asassn_variables_x.csv'))

    period_results = pd.DataFrame({'name':[], 'period':[], 'power':[], 'fap':[], 'actual_period':[], 'actual_class':[]})

    # go through all the lc files in DATA_DIR
    for file in tqdm(os.listdir(DATA_DIR)):
        if file.endswith('.dat'): # make sure it's a valid file\
            time_ls, mag_ls, mag_err_ls = get_time_mag_magerr(file)
            period, power, fap = get_LS_period(time_ls, mag_ls, mag_err_ls, file)

            # also want to get the ground truth values
            name = get_name(file)
            actual_period = vars.loc[vars['name']==name]['period'].values[0]
            actual_class = vars.loc[vars['name']==name]['class'].values[0]

            # add to the dataframe
            period_results = pd.concat([period_results, pd.DataFrame({'name':[name], 'period':[period], 'power':[power], 'fap':[fap], 'actual_period':[actual_period], 'actual_class':[actual_class]})])

    # save the dataframe
    period_results.to_csv(os.path.join(VAR_OUT, 'period_results.csv'), index=False)

    # visualize the results!
    # ideas:
    # 1. compare histogram of the fap values for periodic vs nonperiodic (based on true class -- check original paper for which objects are periodic; or if the true period values are 0 or NaN, then could use that)
    # 2. histogram (or plot calculated period vs actual period) for the periods you calculate vs the actual periods --> break it down based on different classes