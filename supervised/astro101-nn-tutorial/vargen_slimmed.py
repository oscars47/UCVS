# created by @oscars47 and @ghirsch123 summer 2022, updated fall 2022
# creates a class called Variable that computes each of the 13 indices for each lightcurve object passed through

import numpy as np
import pandas as pd
import os
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis

# assumes input is single .dat file for the variable of interest
class Variable:
    # initializing-----------------------------
    def __init__(self, file, name, isper, target, input):
        
        # create dataframe
        c_path = os.path.join(input, file)
        lc_df = pd.read_csv(c_path, sep='\t')
        # initialize time, mag, and mag_err lists
        self.time_ls = np.array(lc_df['HJD'].to_list()).astype(float)
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
        self.mag_ls = np.array(mag_c).astype(float)
        self.mag_err_ls = np.array(lc_df['mag_err'].to_list()).astype(float)

        # set name to be its id
        self.name = name
        # set target type to match the value in the 'ML' column in the ASAS-SN catalog
        self.target = target
        # is variable periodic?
        self.isper = isper

        # initialize each of the var indices
        # median absolute deviation
        self.mad = self.get_mad()
        # weighted mean
        self.weighted_mean = self.get_weighted_mean()
        # reduced chi2
        self.chi2red = self.get_chi2red()
        # weighted standard deviation
        self.weighted_stdev = self.get_weighted_stdev()
        # interquartile range
        self.iqr = self.get_iqr()
        # robust median statistic
        self.roms = self.get_roms()
        # normalized excess variance
        self.norm_excess_var = self.get_norm_excess_var()
        # peak to peak variability
        self.peak_peak_var = self.get_peak_peak_var()
        # von neumann ratio
        self.eta_ratio = self.get_eta_ratio()
        # variability detection statistic
        self.SB = self.get_SB()
        # clipped standard deviation
        self.clipped_std = self.get_clipped_stdev()
        # lomb-scargle period and power
        self.period, self.power = self.get_LS_period()
        self.skew = self.get_skew()
        self.kurt = self.get_kurtosis()
    
    # the lomb-scargle statistic can already be calculated using the astropy package
    def get_LS_period(self):
        try:
            ls = LombScargle(self.time_ls, self.mag_ls, self.mag_err_ls, normalization='standard')
            freq, power = ls.autopower(method='fastchi2')

            best_frequency = freq[np.argmax(power)]
            period = 1/ best_frequency
            
            return period, power.max()
        except:
            print("object %s's period could not be calculated" %self.name)
            return 0,0

    # creating var function---------------------
    def get_mad(self):
        #compute median of self.mag_ls
        median_mag = np.median(self.mag_ls)
        
        #now compute absolute deviations from the median
        absolute_deviations_list = []
        for mag in self.mag_ls:
            absolute_deviations_list.append(abs(mag - median_mag)) 
            
        #compute median of the absolute deviations; this is MAD!
        mad = np.median(absolute_deviations_list)
        
        return mad

    def get_weighted_mean(self):
        #create empty lists to sum at end
        weighted_mean_num_list = []
        weighted_mean_denom_list = []
        for i, magerr in enumerate(self.mag_err_ls):
            weighted_mean_num_list.append(
                self.mag_ls[i] / (magerr**2)
            )
            
            weighted_mean_denom_list.append(
                1 / (magerr**2)
            ) 
        weighted_mean = sum(weighted_mean_num_list) / sum(weighted_mean_denom_list)
        return weighted_mean
    
    def get_chi2red(self):
         #chi2-------
        chi2_list = []
        for i, mag in enumerate(self.mag_ls):
            chi2_list.append(
                ((mag - self.weighted_mean)**2) / (self.mag_err_ls[i]**2)
            )
            
        chi2 = sum(chi2_list)
        
        #chi2 reduced-----
        chi2_red = chi2 / (len(self.mag_ls) - 1)
        return chi2_red
    
    def get_weighted_stdev(self):
        #calculate first term multiplying the sum of the weights*(mag - magerr)^2
        weights_list = []
        square_weights_list = []
        for magerr in self.mag_err_ls:
            weight = 1 / (magerr**2)
            weights_list.append(weight)
            square_weights_list.append(weight**2)
            
        left_term = sum(weights_list) / ((sum(weights_list)**2) - sum(square_weights_list))
        
        right_term_list = []
        for i, mag in enumerate(self.mag_ls):
            right_term_list.append(
                weights_list[i]*((mag - self.weighted_mean)**2)
            )
            
        right_term = sum(right_term_list)
        
        #return square root of left term * right term
        weighted_std_dev = (left_term * right_term)**0.5
        return weighted_std_dev

    #IQR
    def get_iqr(self):
        #use np.percentile to find the value of 25% and 75% in the mag list
        q3, q1 = np.percentile(self.mag_ls, [75, 25])
        iqr = q3 - q1
        return iqr

    #robust median statistic
    #for non-variable object, expected value is around 1
    def get_roms(self):
        median_mag = np.median(self.mag_ls)
        right_term_list = []
        for i, mag in enumerate(self.mag_ls):
            right_term_list.append(
            abs(mag - median_mag) / self.mag_err_ls[i]
            )
            
        right_term = sum(right_term_list)
        left_term = 1 / (len(self.mag_ls) - 1)
        roms = left_term * right_term
        return roms

    #normalized excess variance
    def get_norm_excess_var(self):
        
        right_term_list = []
        for i, mag in enumerate(self.mag_ls):
            right_term_list.append(
                ((mag - self.weighted_mean)**2 - (self.mag_err_ls[i]**2))
            )
            
        right_term = sum(right_term_list)
        left_term = 1 / (len(self.mag_ls)*(self.weighted_mean**2))
        norm_excess_var = left_term * right_term
        return norm_excess_var

    #peak to peak variability
    def get_peak_peak_var(self):
        #create list of mag + magerr and mag - magerr
        mag_plus_magerr_list = []
        mag_minus_magerr_list = []
        for i, mag in enumerate(self.mag_ls):
            mag_plus_magerr_list.append(mag + self.mag_err_ls[i])
            mag_minus_magerr_list.append(mag - self.mag_err_ls[i])
            
        #find max and min for minus, plus respectively
        min_mag_plus_magerr = min(mag_plus_magerr_list)
        max_mag_minus_magerr = max(mag_minus_magerr_list)
        
        peak_to_peak_var = (max_mag_minus_magerr - min_mag_plus_magerr) / (max_mag_minus_magerr + min_mag_plus_magerr)
        return peak_to_peak_var


    #von Neumann ratio: 1 / eta
    def get_eta_ratio(self):
        
        #define numerator and denomimator lists over which to sum and then divide
        num_list = []
        denom_list = []
        
        #for num list
        for i in range(len(self.mag_ls) - 1):
            num_list.append(
                ((self.mag_ls[i+1] - self.mag_ls[i])**2) / (len(self.mag_ls) - 1)
            )
        num = sum(num_list)
        
        #for denom list
        for i, mag in enumerate(self.mag_ls):
            denom_list.append(
                ((mag - self.weighted_mean)**2) / (len(self.mag_ls) - 1)
            )
        denom = sum(denom_list) 
        
        eta = num / denom
        #we want to return 1 / eta as the measure, so take the reciprocal
        if eta != 0:
            return 1 / eta
        else:
            return 

    # helper function
    #sign function        
    def sgn(self, x):
        if x > 0:
            return 1
        else:
            return -1  

    def get_SB(self):
        #first need to find the groups of consecutive same sign residuals; so compute list of residuals
        
        residuals_list = []
        residuals_sign = []
        for i, mag in enumerate(self.mag_ls):
            residual = mag - self.weighted_mean 
            residuals_list.append(residual)
            residuals_sign.append(self.sgn(residual)) 
            
        #list of consecutive same-sign residuals; each element is a list of (residual, error) tuples
        CSSRG_list = []
        #now go through residuals sign list and group based on sign
        i = 0 
        while i < len(residuals_sign):    
            #if we're at the end of the list, then break
            if i==len(residuals_sign)-1:
                break
            
            #if +1, check if other +1
            if residuals_sign[i]==1:
                #temp list holding the (residual, error) tuples
                CSSRG = [(residuals_list[i], self.mag_err_ls[i])]
                for j in range(i+1, len(residuals_sign)):
                    #if we run into a -1, end of consecutive group
                    if residuals_sign[j] == -1:
                        #check to see there are at least 2 objects in group; otherwise, if only one object, can't be a group
                        if len(CSSRG)<2:
                            #don't append CSSRG to CSSRG_list; increment i; need to subract 1 so we start the next group at the dissenting index
                            i += (j-i)-1
                            break
                        #if we have >= 2 residuals in a group, but groups ends, append to CSSRG_list
                        else:
                            CSSRG_list.append(CSSRG)
                            i += (j-i)-1
                            break
                    #same sign!
                    else:
                        CSSRG.append((residuals_list[j], self.mag_err_ls[j]))
                        #if i == max_index-1 and we haven't hit a -1, append CSSRG to main list
                        if j == len(residuals_sign)-1:
                            CSSRG_list.append(CSSRG)
            #if -1, check if other -1
            else:
                #temp list holding the (residual, error) tuples
                CSSRG = [(residuals_list[i], self.mag_err_ls[i])]
                for j in range(i+1, len(residuals_sign)):
                    #if we run into a +1, end of consecutive group
                    if residuals_sign[j] == 1:
                        #check to see there are at least 2 objects in group; otherwise, if only one object, can't be a group
                        if len(CSSRG)<2:
                            #don't append CSSRG to CSSRG_list; increment i
                            i += (j-i)-1
                            break
                        #if we have >= 2 residuals in a group, but groups ends, append to CSSRG_list
                        else:
                            CSSRG_list.append(CSSRG)
                            i += (j-i)-1
                            break
                    #same sign!
                    else:
                        CSSRG.append((residuals_list[j], self.mag_err_ls[j]))
                        #if i == max_index-1 and we haven't hit a -1, append CSSRG to main list
                        if j == len(residuals_sign)-1:
                            CSSRG_list.append(CSSRG)
            i+=1
        
        #with the list of list of (residual, error) tuples, we can actually compute SB!!
        right_term_list = []
        for resid_err_list in CSSRG_list:
            #define temp var to hold sum generated in second for loop
            resid_err_sum = 0
            for resid_err in resid_err_list:
                resid_err_sum += (resid_err[0] / resid_err[1])
            #now append the square of the sum to the right_term_list
            right_term_list.append(
                resid_err_sum**2
            )
        if (len(self.mag_ls) > 1) and (len(CSSRG_list) > 1):
            right_term = sum(right_term_list)
            left_term = 1 / (len(self.mag_ls)*len(CSSRG_list))
            SB = left_term * right_term
        else:
            SB = 0
        return SB

    #clipped mag -- remove top brighest and dimmest sources from lightcurve
    # helper function for clipped_stdev
    
    def get_clipped(self):
        mag_list = list(self.mag_ls.copy()) # get copy of mag_ls so when we remove points it's ok
        #put high mag first (i.e. dimmer)
        dim_ordered = sorted(mag_list, reverse=True)
        #order based on decreasing mag (i.e. brighter)
        bright_ordered = sorted(mag_list)
        
        if (len(mag_list) >= 5) and (len(mag_list) <= 10):
            #remove brightest and dimmest source
            mag_list.remove(dim_ordered[0])
            mag_list.remove(bright_ordered[0])
            
        elif (len(mag_list) > 10) and (len(mag_list) <= 15):
            #drop the first two from dim and bright lists
            for i in range(0, 2):
                mag_list.remove(dim_ordered[i])
                mag_list.remove(bright_ordered[i])
                
        elif (len(mag_list) > 15) and (len(mag_list) <= 20):
            #drop the first three from dim and bright lists
            for i in range(0, 3):
                mag_list.remove(dim_ordered[i])
                mag_list.remove(bright_ordered[i])
                
        elif (len(mag_list) > 20) and (len(mag_list) <= 25):
            #drop the first four from dim and bright lists
            for i in range(0, 4):
                mag_list.remove(dim_ordered[i])
                mag_list.remove(bright_ordered[i])
        
        else:
            #drop the first five from dim and bright lists
            for i in range(0, 5):
                mag_list.remove(dim_ordered[i])
                mag_list.remove(bright_ordered[i])
                
        return mag_list

    #clipped stdev
    def get_clipped_stdev(self):
        #get clipped mag_list
        mag_list = self.get_clipped()
        mean_mag = np.mean(mag_list)
        
        sq_residuals = []
        for mag in mag_list:
            sq_residuals.append(
                (mag - mean_mag)**2
            )
        right_term = sum(sq_residuals)
        left_term = 1 / (len(mag_list)-1)
        stdev = (left_term * right_term)**0.5
        return stdev

    def get_skew(self):
        return skew(self.mag_ls)
    
    def get_kurtosis(self):
        return kurtosis(self.mag_ls)

    #compile all stats into dataframe entry; return dict
    def return_dict(self):
        return {
            'id': [self.name],'target': [self.target], 'periodic': [self.isper],
            'mad': [self.mad], 'weighted mean': [self.weighted_mean],
            'chi2red': [self.chi2red], 'weighted stdev': [self.weighted_stdev], 
            'iqr': [self.iqr], 'roms': [self.roms], 'norm excess var': [self.norm_excess_var], 
            'peak peak var': [self.peak_peak_var], 'eta ratio': [self.eta_ratio], 
            'SB': [self.SB],'clipped stdev': [self.clipped_std], 'period': [self.period], 'power': [self.power],
            'skew': [self.skew], 'kurtosis': [self.kurt]
        }
