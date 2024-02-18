# choose file names using some function
# open corresponding lightcurves, standardize, cut into subseqs
# save each subsequence in a directory

from vari_tree_lib.lightcurve import ASASSN_Lightcurve
from vari_tree_lib.pickle_data import lightly_unpickle
from vari_tree_lib.pickle_data import lightly_pickle
import pandas as pd
import numpy as np
import sys
import os
import matplotlib.pyplot as plt
import configparser
import pickle

parentDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # find our config file
config = configparser.ConfigParser()
config.read(os.path.join(parentDir, "tree_config.txt"))
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
data_dir = config["data_dir"]
pickled_df_dir = config["pickle_df_dir"]
subseq_dir = config["sample_dir"]

time_window = 250
time_step = 10

def fix_name(filename):
    filename = filename.replace(" ", "_") + ".dat"
    filename = os.path.join(data_dir, "g_band_lcs", filename)
    return filename

def choose_files(num_files, min_dpoints = 100, prev = 0):
    if os.path.exists(os.path.join(data_dir,"aws_files.txt")):
        filenames = []
        with open(os.path.join(data_dir,"aws_files.txt"), "rb") as f:
            filenames.append(f.readline().strip())
        rand_files = np.random.choice(filenames, num_files)
    else:
        df = pd.read_csv(os.path.join(data_dir, "asassn_rounded.csv"))
        rand_files = df["ID"].sample(num_files).to_list()
    rand_files = [fix_name(i) for i in rand_files]
    print(rand_files)
    rand_files = [i for i in rand_files if os.path.isfile(i) and len(ASASSN_Lightcurve.from_dat_file(i).times) >= min_dpoints]
    print(rand_files)
    print(len(rand_files))
    if len(rand_files) + prev < num_files:
        rand_files.extend(choose_files(num_files,min_dpoints, len(rand_files)+prev))
    rand_files = rand_files[:num_files]
    return rand_files

# lc = lightly_unpickle(os.path.join(subseq_dir, os.listdir(subseq_dir)[0]))
# lc.plot()
# plt.show()

files = choose_files(50)

# print(files)
# print([type(i) for i in files])
for f in files:
    lc = ASASSN_Lightcurve.from_dat_file(f)
    lc.standardize()
    subseqs = lc.sliding_subseqs(time_window, time_step)
    for i, sbsq in enumerate(subseqs):
        outfile = os.path.join(subseq_dir, f'{ASASSN_Lightcurve.id_from_filename(f)}_{time_window}dw_{time_step}ds_{i}')
        lightly_pickle(sbsq, outfile, protocol=4)