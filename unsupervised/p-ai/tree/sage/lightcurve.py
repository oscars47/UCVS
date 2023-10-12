import math
import os
import random

import matplotlib.colors as mcolors
from tqdm import tqdm
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import astropy
import configparser

config = configparser.ConfigParser()
config.read("../tree_config.txt")
config = config["DEFAULT"]
data_dir = config["data_dir"]
lc_dir = os.path.join(data_dir, "g_band_lcs")

colors = list(mcolors.XKCD_COLORS)

class Lightcurve:
    def __init__(self, time_arr, mag_arr, mag_err_arr, ID, obj_type=None, parent_id=None):
        self.times = time_arr
        self.mags = mag_arr
        self.mag_err = mag_err_arr
        self.id = ID
        self.obj_type = obj_type
        self.parent_id = parent_id
        self.num_subseqs = 0

    @classmethod
    def from_dat_file(cls, path):
        lc_df = pd.read_csv(path, sep='\t')
        time = lc_df['HJD'].astype(np.float32).to_list()
        mag = [float(v.replace(">", '').replace("<", '')) for v in lc_df['mag'].astype(str).to_list()]
        mag_err = lc_df['mag_err'].astype(np.float32).to_list()
        ID = path.split(os.sep)[-1].split('.dat')[0].split('_')
        ID = ID[0] + ' ' + ID[1]
        return cls(time, mag, mag_err, ID=ID)

    @property
    def total_time(self):
        if len(self.times):
            return self.times[-1] - self.times[0]
        return 0

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(self.times, self.mags)
        # plt.errorbar(self.times, self.mags, yerr=self.mag_err, fmt='o', color=random.choice(colors))
        plt.xlabel('Time (HJD)', fontsize=14)
        plt.ylabel('Mag', fontsize=14)
        plt.title('Lightcurve for Object ' + self.id, fontsize=16)
        plt.show()

    def get_subseq(self, start_time, end_time):

        # bounds checking
        if self.times[0] > end_time or self.times[-1] < start_time:
            self.num_subseqs += 1
            return Lightcurve([], [], [], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id, obj_type=self.obj_type)

        # find first and last time indices
        curve_start = None
        curve_end = None
        for i in range(len(self.times)):
            if curve_start is None and self.times[i] > start_time:
                curve_start = i
            if self.times[i] > end_time:
                curve_end = i
                break
        if curve_end is None:
            curve_end = len(self.times)

        sub_times = self.times[curve_start:curve_end]
        sub_mags = self.mags[curve_start:curve_end]
        sub_mag_err = self.mag_err[curve_start:curve_end]
        
        print(f"making subseq for window between {start_time} and {end_time} with {len(sub_times)} data points.")
        self.num_subseqs +=1 
        return Lightcurve(sub_times, sub_mags, sub_mag_err, ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id, obj_type=self.obj_type)

        # gather times in between
        # associate with magnitude and mag errr
        # construct and return lightcurve object



if __name__ == "__main__":
    # "ASASSN-V_J055803.37-143014.8.dat"
    # files = os.listdir(lc_dir)[:100]
    # with open("lightcurves.txt", "w+") as f:
    #     f.write('\n'.join(files))
    with open("lightcurves.txt", "r") as f:
        files = [p.replace("\n", '') for p in f.readlines()]

    files = files[:20]
    for i in tqdm(range(len(files)), desc='plotting lightcurves', position=0, leave=True):
        f = files[i]
        seq = Lightcurve.from_dat_file(os.path.join(lc_dir, f))
        # seq.plot()

    start = seq.times[math.floor(len(seq.times)/4)]
    end = seq.times[math.floor(3*len(seq.times)/4)]

    sub_seq = seq.get_subseq(start, end)
    seq.plot()
    sub_seq.plot()