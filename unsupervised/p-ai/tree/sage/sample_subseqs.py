import math
import os, sys
import pickle
import random
import time

import matplotlib.colors as mcolors
from tqdm import tqdm
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import astropy
import configparser
from vari_tree_lib.lightcurve import ASASSN_Lightcurve
from vari_tree_lib.pickle_data import lightly_pickle, lightly_unpickle

config = configparser.ConfigParser()
config.read("../tree_config.txt")
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
data_dir = config["data_dir"]
sample_dir = config["sample_storage_dir"]
pickled_df_dir = config["pickle_df_dir"]
lc_dir = os.path.join(data_dir, "g_band_lcs")


def test_standardization():
    fig, axes = plt.subplots(2, figsize=(10, 7))
    for i in range(3):
        seq = ASASSN_Lightcurve.from_dat_file(random.choice(filepaths))
        axes[0].scatter(seq.times, seq.mags, color=seq.marker_color, alpha=0.5, s=3)
        seq.standardize()
        axes[1].scatter(seq.times, seq.mags, color=seq.marker_color, alpha=0.5, s=3)

    axes[0].set_title("Unstandardized")
    axes[1].set_title("Standardized")
    fig.show()


if __name__ == "__main__":
    dataset = "ASASSN"
    num_samples = 10  # number of lightcurves to pull initial subseqs from
    time_window = 50  # length of sliding window, days
    time_step = 10  # for later use, now just in filename

    # we're going to make a file to store the names of each of the lightcurve files because indexing the folder everytime is expensive
    if not os.path.exists("all_pickled_lightcurve_names.txt"):
        files = os.listdir(pickled_df_dir)
        with open("all_pickled_lightcurve_names.txt", "w+") as f:
            f.write('\n'.join(files))

    with open("all_lightcurve_names.txt", "r") as f:
        filepaths = [os.path.join(lc_dir, p.replace("\n", '')) for p in f.readlines()]

    filepaths = filepaths[:80000]

    files_to_sample = random.sample(filepaths, int(num_samples*1.25))  # we're going to have to throw some short ones out
    subsequences = []
    times = []
    for i in tqdm(range(len(files_to_sample)), desc='Sampling random subsequences', position=0, leave=True, colour="green"):
        start = time.time()
        f = files_to_sample[i]
        subsequences.append(ASASSN_Lightcurve.from_dat_file(f).random_subsequence(window_duration=time_window))
        times.append(time.time()-start)
    print(f".dat, randomly loading and slicing {len(subsequences)} lightcurves: average {np.mean(times)}s, median {np.median(times)}s, max {np.max(times)}s, min {np.min(times)}s ")
    files_to_sample = [f.replace(".dat",".df.pkl").replace(lc_dir,pickled_df_dir) for f in files_to_sample]
    subsequences = []
    times = []
    for i in tqdm(range(len(files_to_sample)), desc='Sampling random subsequences', position=0, leave=True, colour="green"):
        start = time.time()
        f = files_to_sample[i]
        subsequences.append(ASASSN_Lightcurve.from_pickle(f).random_subsequence(window_duration=time_window))
        times.append(time.time()-start)
    print(f"pickle, randomly loading and slicing {len(subsequences)} lightcurves: average {np.mean(times)}s, median {np.median(times)}s, max {np.max(times)}s, min {np.min(times)}s ")

    subsequences = subsequences[:num_samples]

    # outfile = "C:\\Users\\chell\\PycharmProjects\\UCVS\\unsupervised\\p-ai\\tree\\sage\\sample_pickle_dir\\sample_ASASSN1_1000_50_10.pkl"
    outfile = os.path.join(sample_dir, f'{num_samples}_{dataset}_samples_{time_window}dw_{time_step}ds.pkl')
    with open(outfile, 'wb') as f:
        pickle.dump(subsequences, f, protocol=4)

    subsequences = lightly_unpickle(outfile)

    subsequences[0].plot()

    num_loads = 1000
    times = []
    for i in range(num_loads):
        start = time.time()
        with open(outfile, 'rb') as f:
            subsequences = pickle.load(f)
        times.append(time.time() - start)
    print(f"Load times ({num_loads} iters loading {num_samples} samples): average {np.mean(times)}s, median {np.median(times)}s, max {np.max(times)}s, min {np.min(times)}s ")
