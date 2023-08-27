# file to hold the Variable2 object, which logs all the features we want to extract from the lightcurve
# called by mmgen2.py
# @oscars47
# note: for information on color processing, see mmgen2.py

import os
import numpy as np
import pandas as pd
from astropy.timeseries import LombScargle
from scipy.stats import skew, kurtosis

class Variable2:
    # initializing------------------------------
    def __init__(self, asassn, table, file, name, target, input):
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
        # read in the asassn and table datasets
        self.asassn = asassn
        self.table = table
        # set name to be its id
        self.name = name
        # set target type to match the value in the 'ML' column in the ASAS-SN catalog
        self.target = target

        # set max_T parameter -- max gap in observations in days
        self.max_T = 14

        # initialize each of var indices--- from asassn paper
        self.period, self.power, self.fap = self.get_LS_period()
        #self.log_fap = np.log10(self.fap+1e-9)# add small correction factor so we don't take log of 0
        self.T_t, self.T_p, self.T_2p, self.delta_t, self.delta_p = self.get_LKLS()
        self.j_k, self.h_k = self.get_color()
        self.skew = self.get_skew()
        self.kurt = self.get_kurtosis()
        self.stdev = self.get_stdev()
        self.median = self.get_median()
        self.iqr = self.get_iqr()
        self.mad = self.get_mad()
        self.weighted_mean = self.get_weighted_mean()
        self.vn = self.get_eta_ratio()
        self.ahl = self.get_ahl()

        # from work over summer
        self.weighted_mean = self.get_weighted_mean()
        self.chi2red = self.get_chi2red()
        self.weighted_stdev = self.get_weighted_stdev()
        self.roms = self.get_roms()
        self.norm_excess_var = self.get_norm_excess_var()
        self.peak_peak_var = self.get_peak_peak_var()
        self.lag1_auto = self.get_lag1_auto()
        self.I= self.get_I()
        self.Ifi = self.get_Ifi()
        self.J = self.get_J()
        self.K = self.get_K()
        self.L = self.get_L()
        self.J_time = self.get_J_time()
        self.I_clipped = self.get_I_clipped()
        self.J_clipped = self.get_J_clipped()
        self.CSSD = self.get_CSSD()
        self.Ex = self.get_Ex()
        self.SB = self.get_SB()
        self.clipped_std = self.get_clipped_stdev()

    # creating var functions
    # function to compute Lomb-Scargle period and FAP
    def get_LS_period(self):
        try:
            ls = LombScargle(self.time_ls, self.mag_ls,self. mag_err_ls, normalization='standard')
            freq, power = ls.autopower(method='fastchi2')
            fap = ls.false_alarm_probability(power.max())

            best_frequency = freq[np.argmax(power)]
            period = 1/ best_frequency
            
            return period, power.max(), fap
        except:
            print("object %s's period could not be calculated" %self.name)
            return 0,0,0

    # compute the LKLS stat for a given sorted mag_ls
    def get_T(self, mag_ls):
        mean = np.mean(mag_ls)
        num_ls = []
        for i in range(len(mag_ls)-1):
            num_ls.append((mag_ls[i+1] - mag_ls[i])**2)
        num = np.sum(num_ls)
        #print('num', num)

        denom_ls = []
        for mag in mag_ls:
            denom_ls.append((mag - mean)**2)
        denom = np.sum(denom_ls)
        #print('denom', denom)

        N = len(mag_ls)
        return (num / denom) * ((N - 1) / (2*N))

    # helper function to sort mag based on folded lc
    def sort_fold(self, period):
        # get folded times
        # need to check if period = 0. if yes, then don't mod period
        if period > 0:
            time_f = []
            for time in self.time_ls:
                time_f.append(time % period)
            
            # now combine into list of tuples
            time_mag = list(zip(time_f, self.mag_ls))
            #print('unsorted', time_mag)
            # now sort based on modded time, which is 0th index
            time_mag.sort(key=lambda x:x[0])
            #print('sorted', time_mag)
            unzipped= list(zip(*time_mag))
            s_time, s_mag = unzipped[0], unzipped[1]
            return s_mag # return the sorted mag
        else:
            return self.time_ls

    # lafler kinmann string length statistic
    def get_LKLS(self):
        # compute T for time
        T_t = self.get_T(self.mag_ls)

        # sort mag_ls based on period and twice period
        s1_mag = self.sort_fold(self.period)
        s2_mag = self.sort_fold(2*self.period)
        # T for period
        T_p = self.get_T(s1_mag)
        T_2p = self.get_T(s2_mag)

        # compute delta_t and delta_p
        delta_t = (T_p - T_t) / T_t
        delta_p = (T_2p - T_p) / T_p

        return T_t, T_p, T_2p, delta_t, delta_p
    
    # function to compute color-----------------
    def get_color(self):
        name_df = self.asassn.loc[self.asassn['ID']==self.name]
        # get ra and dec; round to 5 decimal places
        as_ra = name_df['rounded ra'].to_list()[0]
        as_dec = name_df['rounded dec'].to_list()[0]
        #print(as_ra, as_dec)
        
        try:
            # find in data_df
            table_row = self.table.loc[(self.table['rounded ra']-as_ra <= 0.001) & (self.table['rounded dec']-as_dec <= 0.001)]
            # get values
            j = table_row['j_m'].to_list()[0]
            h = table_row['h_m'].to_list()[0]
            k = table_row['k_m'].to_list()[0]

            return j-k, h-k

        except:
            print('ID %s could not be found in the table dataset' %self.name)
            return 0, 0
        
    # skew and kurtosis
    def get_skew(self):
        return skew(self.mag_ls)
    
    def get_kurtosis(self):
        return kurtosis(self.mag_ls)
    
    #stdev, median, iqr, mad
    def get_stdev(self):
        return np.std(self.mag_ls)
    
    def get_median(self):
        return np.median(self.mag_ls)
    
    def get_iqr(self):
        #use np.percentile to find the value of 25% and 75% in the mag list
        q3, q1 = np.percentile(self.mag_ls, [75, 25])
        iqr = q3 - q1
        return iqr
    
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
    
    
    # inverse von neumann
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
        
    # A_hl (ratio of mag higher/lower vs mean mag
    def get_ahl(self):
        mean = np.mean(self.mag_ls)
        higher = []
        lower = []
        for mag in self.mag_ls:
            if mag > mean:
                higher.append(mag)
            else:
                lower.append(mag)
        
        # return ratio of sizes
        return len(higher) / len(lower)
    

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

    #Lag-1 autocorrection
    def get_lag1_auto(self):
        num_list = []
        for i in range(len(self.mag_ls)-1):
            num_list.append(
                (self.mag_ls[i] - self.weighted_mean) * (self.mag_ls[i+1] - self.weighted_mean)
            )
        
        num = sum(num_list)
            
        denom_list = []
        for mag in self.mag_ls:
            denom_list.append(
                (mag - self.weighted_mean)**2
            )
            
        denom = sum(denom_list)
        
        lag1_auto = num / denom
        return lag1_auto

    # Welsch-Stetson var index
    def get_I(self):
        # choose max time interval between consecutive measurments
        #if only 1 filter, assign odd indices to one sample, even to another
        #but use f1_mean_mag = f2_mean_mag over all the N measurements
        
        #create list of tuples (time, mag, magerr) to sort by time (0th index)
        time_mag_magerr_list = list(zip(self.time_ls, self.mag_ls, self.mag_err_ls))
        
        #print(time_mag_magerr_list)
        
        #check if list has even or odd length; if odd, discard final measurement
        if len(time_mag_magerr_list) % 2 != 0:
            time_mag_magerr_list.pop(-1)
        
        #now iterate through and append to even/odd lists by index
        even_list = []
        odd_list = []
        for i in range(len(time_mag_magerr_list)):
            #even
            if i%2 == 0:
                even_list.append(time_mag_magerr_list[i])
            else:
                odd_list.append(time_mag_magerr_list[i])
        
        #confirm time differences in measurements < max_T
        i = 0
        max_index = len(even_list)
        while i < max_index:
            #first check if the length of the list > 3
            if (len(even_list) <= 3) or (len(odd_list) <= 3):
                break
            
            #if not, then remove the pair
            elif abs(even_list[i][0] - odd_list[i][0]) > self.max_T:
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index -= 1
                
            else:
                i+=1
                
        #now we can calculate I!
        right_term_list = []
        for i in range(len(even_list)):
            right_term_list.append(
                ((even_list[i][1] - self.weighted_mean) / even_list[i][2]) * ((odd_list[i][1] - self.weighted_mean) / odd_list[i][2])
            )
            
        right_term = sum(right_term_list)
        left_term = (1 / (len(even_list)*(len(even_list)-1)))**0.5
        I = left_term * right_term
        return I

    # helper function
    #sign function
    def sgn(self, x):
        if x > 0:
            return 1
        else:
            return -1  
                
    def get_Ifi(self):
        #if only 1 filter, assign odd indices to one sample, even to another
        #but use f1_mean_mag = f2_mean_mag over all the N measurements

        #create list of tuples (time, mag, magerr) to sort by time (0th index)
        time_mag_magerr_list = list(zip(self.time_ls, self.mag_ls, self.mag_err_ls))
        
        #print(time_mag_magerr_list)
        
        #check if list has even or odd length; if odd, discard final measurement
        if len(time_mag_magerr_list) % 2 != 0:
            time_mag_magerr_list.pop(-1)
        
        #now iterate through and append to even/odd lists by index
        even_list = []
        odd_list = []
        for i in range(len(time_mag_magerr_list)):
            #even
            if i%2 == 0:
                even_list.append(time_mag_magerr_list[i])
            else:
                odd_list.append(time_mag_magerr_list[i])
        
        #confirm time differences in measurements < max_T
        i = 0
        max_index = len(even_list)
        while i < max_index:
            #first check if the length of the list > 3
            if (len(even_list) <= 3) or (len(odd_list) <= 3):
                break
            
            #if not, then remove the pair
            elif abs(even_list[i][0] - odd_list[i][0]) > self.max_T:
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index -= 1
                
            else:
                i+=1
                
        #now we can calculate I!
        right_term_list = []
        for i in range(len(even_list)):
            right_term_list.append(
                self.sgn((even_list[i][1] - self.weighted_mean) / even_list[i][2]) * self.sgn((odd_list[i][1] - self.weighted_mean) / odd_list[i][2])
            )
            
        right_term = sum(right_term_list)
        
        Ifi= (1 / (len(even_list))* right_term + 1) *0.5
        return Ifi

    # Stetson's J
    def get_J(self):
        #if only 1 filter, assign odd indices to one sample, even to another
        #but use f1_mean_mag = f2_mean_mag over all the N measurements
        #print(lightcurve_data_all[0][2])
        
        #before sorting, create list of tuples
        time_mag_magerr_list = list(zip(self.time_ls, self.mag_ls, self.mag_err_ls))
        
        #check if list has even or odd length; if odd, add to individual measurement
        inidividual_time_mag_magerr_list = []
        if len(time_mag_magerr_list) % 2 != 0:
            inidividual_time_mag_magerr_list.append(time_mag_magerr_list[-1])
            time_mag_magerr_list.pop(-1)
        
        #now iterate through and append to even/odd lists by index
        even_list = []
        odd_list = []
        for i in range(len(time_mag_magerr_list)):
            #even
            if i%2 == 0:
                even_list.append(time_mag_magerr_list[i])
            else:
                odd_list.append(time_mag_magerr_list[i])
        
        #confirm time differences in measurements < max_T
        i = 0
        max_index = len(even_list)
        while i < max_index:
            #first check if the length of the list > 3
            if (len(even_list) <= 3) or (len(odd_list) <= 3):
                break
            
            #if not, then remove the pair from pair list but add to individual list
            elif abs(even_list[i][0] - odd_list[i][0]) > self.max_T:
                inidividual_time_mag_magerr_list.append(even_list[i])
                inidividual_time_mag_magerr_list.append(odd_list[i])
                
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index -= 1
                
            else:
                i+=1
                
        #now we can calculate J!
        J_num_list = []
        J_denom_list = []
        #first consider the pairs---
        for i in range(len(even_list)):
            Pk_left_left = ((len(even_list)) / (len(even_list) - 1))**0.5
            Pk_left_right = ((even_list[i][1] - self.weighted_mean) / even_list[i][2])
            Pk_right_left = ((len(odd_list)) / (len(odd_list) - 1))**0.5
            Pk_right_right = ((odd_list[i][1] - self.weighted_mean) / odd_list[i][2])
            Pk = Pk_left_left * Pk_left_right * Pk_right_left * Pk_right_right

            #get weight: median of individual weights
            weight = np.median([1 / (even_list[i][2]**2), 1 / (odd_list[i][2] **2)])

            #append to num and denom lists
            J_num_list.append(
                weight * self.sgn(Pk) * (abs(Pk))**0.5
            )

            J_denom_list.append(
                weight
            )
        #now consider the inidividual measurements
        if len(inidividual_time_mag_magerr_list) > 1:
            for i in range(len(inidividual_time_mag_magerr_list)):
                Pk_left = len(inidividual_time_mag_magerr_list) / (len(inidividual_time_mag_magerr_list) - 1) 
                Pk_right = ((inidividual_time_mag_magerr_list[i][1] - self.weighted_mean) / inidividual_time_mag_magerr_list[i][2])**2
                Pk = Pk_left * Pk_right - 1

                #weight: that for first filter
                weight = 1 / (inidividual_time_mag_magerr_list[i][2]**2)

                #append to num and denom lists
                J_num_list.append(
                    weight * self.sgn(Pk) * (abs(Pk))**0.5
                )

                J_denom_list.append(
                    weight
                )


        #now get complete J_num and J_denom
        J_num = sum(J_num_list)
        J_denom = sum(J_denom_list)
        if J_denom != 0:
            J = J_num / J_denom
        else:
            J = 0
        return J

    # Stetson's K
    def get_K(self):
        #compute K!
        K_num_list = []  
        K_denom_list = []
        #total number of observations:
        N = len(self.mag_ls)
        
        for j, mag in enumerate(self.mag_ls):
            left_term = (len(self.mag_ls) / (len(self.mag_ls) - 1))**0.5
            right_term = (mag - self.weighted_mean) / self.mag_err_ls[j]
            #append to K_num_list
            K_num_list.append(
                abs(left_term * right_term)
            )
            #append to K_denom_list
            K_denom_list.append(
                (left_term * right_term)**2
            )
           
        K_complete_num = (1 / N) * sum(K_num_list)
        K_complete_denom = ((1 / N) * sum(K_denom_list))**0.5
        K = K_complete_num / K_complete_denom
        return K

    # Stetson's L
    def get_L(self):
        L = ((np.pi / 2)**0.5)*self.J*self.K
        return L

    # Stetson's J but with time weighting
    #we use all of the above code for J, except we forgo a delta T parameter and instead weight using the time differneces
    #i.e., w_i = exp(-(t_(i+1) - t_i) / delta t), where delta t is the median of all the pairs (t_(i+1) - t_i)
    def get_J_time(self):
        #if only 1 filter, assign odd indices to one sample, even to another
        #but use f1_mean_mag = f2_mean_mag over all the N measurements
        #print(lightcurve_data_all[0][2])
        
        #before sorting, create list of tuples
        time_mag_magerr_list = list(zip(self.time_ls, self.mag_ls, self.mag_err_ls))
        
        #check if list has even or odd length; if odd, add to individual measurement
        inidividual_time_mag_magerr_list = []
        if len(time_mag_magerr_list) % 2 != 0:
            inidividual_time_mag_magerr_list.append(time_mag_magerr_list[-1])
            time_mag_magerr_list.pop(-1)
        
        #now iterate through and append to even/odd lists by index
        even_list = []
        odd_list = []
        for i in range(len(time_mag_magerr_list)):
            #even
            if i%2 == 0:
                even_list.append(time_mag_magerr_list[i])
            else:
                odd_list.append(time_mag_magerr_list[i])
        
        #confirm time differences in measurements < max_T
        i = 0
        max_index = len(even_list)
        while i < max_index:
            #first check if the length of the list >= 3
            if (len(even_list) < 3) or (len(odd_list) < 3):
                break
            
            #if not, then remove the pair from pair list but add to individual list
            elif abs(even_list[i][0] - odd_list[i][0]) > self.max_T:
                inidividual_time_mag_magerr_list.append(even_list[i])
                inidividual_time_mag_magerr_list.append(odd_list[i])
                
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index-=1
                
            else:
                i+=1
                
        #now we can calculate J!
        J_num_list = []
        J_denom_list = []
        #first consider the pairs---
        for i in range(len(even_list)):
            Pk_left_left = ((len(even_list)) / (len(even_list) - 1))**0.5
            Pk_left_right = ((even_list[i][1] - self.weighted_mean) / even_list[i][2])
            Pk_right_left = ((len(odd_list)) / (len(odd_list) - 1))**0.5
            Pk_right_right = ((odd_list[i][1] - self.weighted_mean) / odd_list[i][2])
            Pk = Pk_left_left * Pk_left_right * Pk_right_left * Pk_right_right

            #get weight: median of individual weights
            weight = np.median([1 / (even_list[i][2]**2), 1 / (odd_list[i][2] **2)])

            #append to num and denom lists
            J_num_list.append(
                weight * self.sgn(Pk) * (abs(Pk))**0.5
            )

            J_denom_list.append(
                weight
            )
        #now consider the inidividual measurements
        if len(inidividual_time_mag_magerr_list) > 1:
            for i in range(len(inidividual_time_mag_magerr_list)):
                Pk_left = len(inidividual_time_mag_magerr_list) / (len(inidividual_time_mag_magerr_list) - 1) 
                Pk_right = ((inidividual_time_mag_magerr_list[i][1] - self.weighted_mean) / inidividual_time_mag_magerr_list[i][2])**2
                Pk = Pk_left * Pk_right - 1

                #weight: that for first filter
                weight = 1 / (inidividual_time_mag_magerr_list[i][2]**2)

                #append to num and denom lists
                J_num_list.append(
                    weight * self.sgn(Pk) * (abs(Pk))**0.5
                )

                J_denom_list.append(
                    weight
                )
            

        #now get complete J_num and J_denom
        J_num = sum(J_num_list)
        J_denom = sum(J_denom_list)
        if J_denom != 0:
                J = J_num / J_denom
        else:
            J = 0
        return J 

    # I clipped
    #like I, except we impose in addition to the time constraint that the pairs' mag must not differ by > 5x combined uncertainty
    def get_I_clipped(self):
        #create list of tuples (time, mag, magerr) to sort by time (0th index)
        time_mag_magerr_list = list(zip(self.time_ls, self.mag_ls, self.mag_err_ls))
        
        #print(time_mag_magerr_list)
        
        #check if list has even or odd length; if odd, discard final measurement
        if len(time_mag_magerr_list) % 2 != 0:
            time_mag_magerr_list.pop(-1)
        
        #now iterate through and append to even/odd lists by index
        even_list = []
        odd_list = []
        for i in range(len(time_mag_magerr_list)):
            #even
            if i%2 == 0:
                even_list.append(time_mag_magerr_list[i])
            else:
                odd_list.append(time_mag_magerr_list[i])
        
        #confirm time differences in measurements < max_T
        i = 0
        max_index = len(even_list)
        while i < max_index:
            #first check if the length of the list > 3
            if (len(even_list) <= 3) or (len(odd_list) <= 3):
                break
            
            #if not, then remove the pair
            elif abs(even_list[i][0] - odd_list[i][0]) > self.max_T:
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index =- 1
            
            #now check for magnitude uncertainty
            elif abs(even_list[i][1] - odd_list[i][1]) > 5*(even_list[i][2] + odd_list[i][2]):
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index =- 1
            
            else:
                i+=1
                
        #now we can calculate I!
        right_term_list = []
        for i in range(len(even_list)):
            right_term_list.append(
                ((even_list[i][1] - self.weighted_mean) / even_list[i][2]) * ((odd_list[i][1] - self.weighted_mean) / odd_list[i][2])
            )
            
        right_term = sum(right_term_list)
        left_term = (1 / (len(even_list)*(len(even_list)-1)))**0.5
        I = left_term * right_term
        return I

    # J clipped
    #same as J, but with same delta mag restriction as with I clipped
    def get_J_clipped(self):
         #if only 1 filter, assign odd indices to one sample, even to another
        #but use f1_mean_mag = f2_mean_mag over all the N measurements
        #print(lightcurve_data_all[0][2])
        
        #before sorting, create list of tuples
        time_mag_magerr_list = list(zip(self.time_ls, self.mag_ls, self.mag_err_ls))
        
        #check if list has even or odd length; if odd, add to individual measurement
        inidividual_time_mag_magerr_list = []
        if len(time_mag_magerr_list) % 2 != 0:
            inidividual_time_mag_magerr_list.append(time_mag_magerr_list[-1])
            time_mag_magerr_list.pop(-1)
            
        #now iterate through and append to even/odd lists by index
        even_list = []
        odd_list = []
        for i in range(len(time_mag_magerr_list)):
            #even
            if i%2 == 0:
                even_list.append(time_mag_magerr_list[i])
            else:
                odd_list.append(time_mag_magerr_list[i])
        
        #confirm time differences in measurements < max_T
        i = 0
        max_index = len(even_list)
        while i < max_index:
            #first check if the length of the list > 3
            if (len(even_list) <= 3) or (len(odd_list) <= 3):
                break
            
            #if not, then remove the pair from pair list but add to individual list
            elif abs(even_list[i][0] - odd_list[i][0]) > self.max_T:
                inidividual_time_mag_magerr_list.append(even_list[i])
                inidividual_time_mag_magerr_list.append(odd_list[i])
                
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index -= 1
                
            #check to confirm delta mag is within 5*combined uncertainty
            elif abs(even_list[i][1] - odd_list[i][1]) > 5*(even_list[i][2] + odd_list[i][2]):
                inidividual_time_mag_magerr_list.append(even_list[i])
                inidividual_time_mag_magerr_list.append(odd_list[i])
                
                even_list.pop(i)
                odd_list.pop(i)
                
                max_index -= 1
            
            else:
                i+=1
                
        #now we can calculate J!
        J_num_list = []
        J_denom_list = []
        #first consider the pairs---
        for i in range(len(even_list)):
            Pk_left_left = ((len(even_list)) / (len(even_list) - 1))**0.5
            Pk_left_right = ((even_list[i][1] - self.weighted_mean) / even_list[i][2])
            Pk_right_left = ((len(odd_list)) / (len(odd_list) - 1))**0.5
            Pk_right_right = ((odd_list[i][1] - self.weighted_mean) / odd_list[i][2])
            Pk = Pk_left_left * Pk_left_right * Pk_right_left * Pk_right_right

            #get weight: median of individual weights
            weight = np.median([1 / (even_list[i][2]**2), 1 / (odd_list[i][2] **2)])

            #append to num and denom lists
            J_num_list.append(
                weight * self.sgn(Pk) * (abs(Pk))**0.5
            )

            J_denom_list.append(
                weight
            )
        #now consider the inidividual measurements
        if len(inidividual_time_mag_magerr_list) > 1:
            for i in range(len(inidividual_time_mag_magerr_list)):
                Pk_left = len(inidividual_time_mag_magerr_list) / (len(inidividual_time_mag_magerr_list) - 1) 
                Pk_right = ((inidividual_time_mag_magerr_list[i][1] - self.weighted_mean) / inidividual_time_mag_magerr_list[i][2])**2
                Pk = Pk_left * Pk_right - 1

                #weight: that for first filter
                weight = 1 / (inidividual_time_mag_magerr_list[i][2]**2)

                #append to num and denom lists
                J_num_list.append(
                    weight * self.sgn(Pk) * (abs(Pk))**0.5
                )

                J_denom_list.append(
                    weight
            )
            

        #now get complete J_num and J_denom
        J_num = sum(J_num_list)
        J_denom = sum(J_denom_list)
        if J_denom != 0:
            J = J_num / J_denom
        else:
            J = 0
        return J 

    #consecutive same-sign deviations from the median mag
    #number of groups with 3 consecutive measurements brighter/fainter than median mag by >= 3*sigma,
    #where sigma is MAD scaled to sigma (sigma = 1.4826 * MAD)
    def get_CSSD(self):
        #compute median of self.mag_ls
        median_mag = np.median(self.mag_ls)
        
        #define sigma
        sigma = 1.4826 * self.mad
        
        #print('sigma', sigma)
        #print('3sigma', 3* sigma)
        
        #find number of consecutive groups with 3 measurements with abs(mag - median) >= 3*sigma
        #initialize CSSD
        CSSD = 0
        
        #iterate through each element in the mag list and compare difference to median
        i = 0
        #use a while loop so we can update the counter to move beyond elements we've already checked
        while i < len(self.mag_ls):
            #if we have a significant deviation from median, use for loop to check remaining elements
            #print('mag diff', abs(self.mag_ls[i] - median_mag))
            if abs(self.mag_ls[i] - median_mag) > 3*sigma:
                #temp to hold how many significant deviations in this round
                N = 1
                #start the for loop from the next index
                for j in range(i+1, len(self.mag_ls)):
                    #print('for diff', abs(self.mag_ls[j] - median_mag))
                    
                    #if this deviation is significant, increment N
                    if abs(self.mag_ls[j] - median_mag) > 3*sigma:
                        #print('suc', abs(self.mag_ls[j] - median_mag))
                        N+=1
                    #if we hit a non-significant deviation, increment i and break since CSSD must be consecutive
                    else:
                    #update i by difference of indices
                        i+= (j-i)
                        #break out of this for loop, back into the while loop
                        break 
                    
                    #check if we've hit N=3
                    if N==3:
                        #print('increment')
                        #increment CSSD by 1
                        CSSD +=1
                        #update i by difference of indices
                        i+= (j-i)
                        #break out of this for loop, back into the while loop
                        break
                #if we make it here, then we never hit N==3; update counter, which will then trigger us to exit the while loop
                i+= len(self.mag_ls)
        
            #increment counter
            i+=1
        
        #need to normalize the number of groups by (N-2), where N is number of pts on lightcurve
        CSSD_normalized = CSSD / (len(self.mag_ls) - 2)
        
        #return the value of CSSD_normalized
        return CSSD_normalized

    #excursions
    def get_Ex(self):
        #now divide self.mag_ls into groups based on max_T
        #define list to hold sub-lists of mag tuples based on time
        groups_list = []
        #define temp list to hold mag for individual group; start it with the first mag
        group = [self.mag_ls[0]]

        for i in range(len(self.mag_ls)-1):
            #if the time difference is within max allowed time, save the next element to the group
            if self.time_ls[i+1] - self.time_ls[i] <= self.max_T:
                group.append(self.mag_ls[i+1])
                #need to check if we're at the end of the list; if so, save the group!
                if i == len(self.mag_ls) - 2:
                    groups_list.append(group)
            
            #if not, then don't append the next element; add group to groups_list, restart group with i+1 mag
            else:
                groups_list.append(group)
                group = [self.mag_ls[i+1]]
                #if i+1 = len(self.mag_ls) - 1, then save this final group
                if i == len(self.mag_ls) -2:
                    #print(self.mag_ls[i+1])
                    groups_list.append(group)
                
        #compute median for each group; find mad for each group to find sigma for each group
        group_medians = []
        group_sigma = []
        for group in groups_list:
            median_mag = np.median(group)
            group_medians.append(median_mag)
            
            #MAD calculation
            #find absolute deviations
            abs_dev = []
            for mag in group:
                abs_dev.append(
                    abs(mag - median_mag)
                )
            mad = np.median(abs_dev)
            sigma = 1.4826 * mad
            group_sigma.append(sigma)
        
        #now can compute terms to sum
        Ex_right_list = []
        for i in range(len(groups_list)-1):
            num = abs(group_medians[i] - group_medians[i+1])
            denom = ((group_sigma[i]**2) + (group_sigma[i+1]**2))**0.5
            if denom != 0:
                Ex_right_list.append(num / denom)
            else:
                Ex_right_list.append(0)
        Ex_right_term = sum(Ex_right_list)
        
        #also get the left term
        if len(groups_list) > 1:
            Ex_left_term = 2 / (len(groups_list)*(len(groups_list) - 1))
            Ex = Ex_left_term * Ex_right_term
        else:
            Ex = 0
        return Ex
    
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

    
    # function to return indices as dictionary
    def return_dict(self):
        return {
            'name': [self.name], 'target': [self.target], 'period': [self.period], 'power': [self.power],
            'T_t': [self.T_t], 'T_p': [self.T_p], 'T_2p': [self.T_2p], 
            'delta_t': [self.delta_t], 'delta_p': [self.delta_p],
            'j-k': [self.j_k], 'h-k': [self.h_k], 'skew': [self.skew],
            'kurtosis': [self.kurt], 'stdev': [self.stdev], 'median': [self.median],
            'iqr': [self.iqr], 'mad': [self.mad], 'von neumann': [self.vn],
            'a_hl': [self.ahl],
            'weighted mean': [self.weighted_mean],
            'chi2red': [self.chi2red], 'weighted stdev': [self.weighted_stdev],
            'roms': [self.roms], 'norm excess var': [self.norm_excess_var], 
            'peak peak var': [self.peak_peak_var], 'lag1 auto': [self.lag1_auto],
            'I': [self.I],'Ifi': [self.Ifi], 'J': [self.J], 'K': [self.K], 'L': [self.L],
            'J time': [self.J_time], 'I clipped': [self.I_clipped],
            'J clipped': [self.J_clipped],
            'Ex': [self.Ex], 'SB': [self.SB],'clipped stdev': [self.clipped_std]
        }
