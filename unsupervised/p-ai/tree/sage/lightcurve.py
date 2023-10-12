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
    def __init__(self, time_arr, data_arr, mag_err_arr, ID, obj_type=None, parent_id=None):
        self.times = time_arr
        self.data = data_arr
        self.mag_err = mag_err_arr
        self.id = ID
        self.obj_type = obj_type
        self.parent_id = parent_id

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
        return self.times[-1] - self.times[0]

    def plot(self):
        plt.figure(figsize=(10, 5))
        plt.scatter(self.times, self.data)
        plt.errorbar(self.times, self.data, yerr=self.mag_err, fmt='o', color=random.choice(colors))
        plt.xlabel('Time (HJD)', fontsize=14)
        plt.ylabel('Mag', fontsize=14)
        plt.title('Lightcurve for Object ' + self.id, fontsize=16)
        plt.show()

    def get_subseq(self,start_time,end_time):



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
        seq.plot()
