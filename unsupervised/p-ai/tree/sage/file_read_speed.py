import math
import os
import random
import time
import timeit
from pandas import HDFStore
from line_profiler_pycharm import profile
import matplotlib.colors as mcolors
from tqdm import tqdm
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import astropy
import configparser
from vari_tree_lib.lightcurve import ASASSN_Lightcurve
from pickle_data import lightly_unpickle

config = configparser.ConfigParser()
config.read("../tree_config.txt")
config = config["DEFAULT"]
data_dir = config["data_dir"]  # specify the path to the "g_band" folder of the UCVS data in your config file
pickled_df_dir = config["pickle_df_dir"]
lc_dir = os.path.join(data_dir, "g_band_lcs")

with open("all_lightcurve_names.txt", "r") as f:
    filepaths = [os.path.join(lc_dir, p.replace("\n", '')) for p in f.readlines()]


outpath = "hdf_test/"
hdf_path = os.path.join(outpath, 'data.h5')
files = filepaths[:20000]
IDs = [ASASSN_Lightcurve.id_from_filename(f) for f in files]

# hdf = HDFStore(hdf_path, "a")
# print("Opened")
# for path, ID in zip(files, IDs):
#     df = pd.read_csv(path, sep='\t', usecols=["HJD", "mag", "mag_err"])
#     print(ID)
#     hdf.put(ID, df)
# hdf.close()

path = files[0]
ID = ASASSN_Lightcurve.id_from_filename(path)

print("testing")
num_repeats = 1000

file_series = [100, 500, 1000, 5000, 10000]
# file_series = [10, 100]
csv_multi_read_times = []
hdf_store_multi_read_times = []
df_hdf_multi_read_times = []
df_hdf_store_multi_read_times = []
pickle_multi_read_times = []

pickled_files = [os.path.join(pickled_df_dir,f) for f in os.listdir(pickled_df_dir)]
for num_files in file_series:
    files = pickled_files[:num_files]
    start = time.time()
    for path in files:
        lightly_unpickle(path)

    pickled_multi_read = (time.time()-start)/num_files
    pickle_multi_read_times.append(pickled_multi_read)
    print(f"{num_files} files - Pickled df:", pickled_multi_read)

for num_files in file_series:
    files = filepaths[:num_files]
    start = time.time()
    for path in files:
        pd.read_csv(path, sep='\t', usecols=["HJD", "mag", "mag_err"])

    csv_multi_read = (time.time()-start)/num_files
    csv_multi_read_times.append(csv_multi_read)
    print(f"{num_files} files - CSV:", csv_multi_read)

hdf = HDFStore(hdf_path, "r")
for num_files in file_series:
    ids = IDs[:num_files]
    start = time.time()
    for ID in ids:
        hdf.get(ID)
    hdf_multi_read = (time.time()-start)/num_files
    hdf_store_multi_read_times.append(hdf_multi_read)
    print(f"{num_files} files - HDF store:", hdf_multi_read)
hdf.close()

for num_files in file_series:
    ids = IDs[:num_files]
    start = time.time()
    for ID in ids:
        pd.read_hdf(hdf_path, key=ID)
    df_hdf_multi_read = (time.time()-start)/num_files
    df_hdf_multi_read_times.append(df_hdf_multi_read)
    print(f"{num_files} files - df HDF:", df_hdf_multi_read)

hdf = HDFStore(hdf_path, "r")
for num_files in file_series:
    ids = IDs[:num_files]
    start = time.time()
    for ID in ids:
        pd.read_hdf(hdf, key=ID)
    df_hdf_store_multi_read = (time.time()-start)/num_files
    df_hdf_store_multi_read_times.append(df_hdf_store_multi_read)
    print(f"{num_files} files - df HDF store:", df_hdf_store_multi_read)
hdf.close()

plt.scatter(file_series, csv_multi_read_times, color="blue", label="csv")
plt.scatter(file_series, hdf_store_multi_read_times, color="red", label="hdf5 store")
plt.scatter(file_series, df_hdf_multi_read_times, color="purple", label="df hdf5")
plt.scatter(file_series, df_hdf_store_multi_read_times, color="orange", label="df hdf5 store")
plt.scatter(file_series, pickle_multi_read_times, color="green", label="pickled dataframes")
plt.legend(title="filetype")
plt.xlabel("Number of reads")
plt.ylabel("Time per read (s)")
plt.title("I/O Comparison")
plt.show()

# num_loads = 1000
# times = []
# for i in range(num_loads):
#     start = time.time()
#     with open(outfile, 'rb') as f:
#         subsequences = pickle.load(f)
#     times.append(time.time() - start)
# print(f"Load times: average {np.mean(times)}s, median {np.median(times)}s, max {np.max(times)}s, min {np.min(times)}s ")
