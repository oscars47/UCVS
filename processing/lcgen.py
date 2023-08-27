# file to generate lightcurves from the ASAS-SN dataset
# n.b. "folding" is a technique where we take take the time values and divide them by the period of the object and take the remainder; this allows us to see the lightcurve in a more compact way, and is useful for periodic objects 

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

# reads in .dat file along with output directory
# target is what the class label in the ASAS-SN dataset is
def get_lc(file, name, target, input, output):
    if file.endswith('.dat'):
        #try:
        # create dataframe
        c_path = os.path.join(input, file)
        lc_df = pd.read_csv(c_path, sep='\t')
        
        # the columns we want are time, mag and mag err
        time = lc_df['HJD'].to_list()
        mag = lc_df['mag'].to_list()
        mag_err = lc_df['mag_err'].to_list()
        
        if not(max(mag_err) > 5):
            # define figure
            plt.figure(figsize=(10, 5))
            plt.scatter(time, mag)
            plt.errorbar(time, mag, yerr=mag_err, fmt='o', color='purple')
            plt.xlabel('Time (HJD)', fontsize=14)
            plt.ylabel('Mag', fontsize=14)
            plt.title('Lightcurve for Object ' + name, fontsize=16)
            plt.savefig(output+'/'+target+'lc_%s.jpeg'%name)
            plt.close()
        else:
            print('err: max err value %f exceeds 5 mag limit'%max(mag_err))
        # except:
        #     print('lightcurve %s could not be created.' %file)
    else:
        print('error. must load .dat file. your file is:', file)

def get_lc_fold(file, name, target, input, output, period):
    if file.endswith('.dat'):
        #try:
        # create dataframe
        c_path = os.path.join(input, file)
        lc_df = pd.read_csv(c_path, sep='\t')
        
        # the columns we want are time, mag and mag err
        time = np.array(lc_df['HJD'].to_list()) # convert to numpy array to fold
        time = time % period
        time = list(time)
        mag = lc_df['mag'].to_list()
        mag_err = lc_df['mag_err'].to_list()
        
        if not(max(mag_err) > 5):
            # define figure
            plt.figure(figsize=(10, 5))
            plt.scatter(time, mag)
            plt.errorbar(time, mag, yerr=mag_err, fmt='o', color='purple')
            plt.xlabel('Time (HJD)', fontsize=14)
            plt.ylabel('Mag', fontsize=14)
            plt.title('Folded Lightcurve for Object ' + name, fontsize=16)
            plt.savefig(output+'/'+target+'lc_folded%s.jpeg'%name)
            plt.close()
        else:
            print('err: max err value %f exceeds 5 mag limit'%max(mag_err))
        # except:
        #     print('lightcurve %s could not be created.' %file)
    else:
        print('error. must load .dat file. your file is:', file)

    
