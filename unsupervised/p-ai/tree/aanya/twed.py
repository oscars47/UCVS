import numpy as np
import pandas as pd
import os
import sys
import configparser
import time
from random import randint

sys.path.append('../')
from sage.vari_tree_lib.lightcurve import ASASSN_Lightcurve
from sage.vari_tree_lib.lightcurve import Lightcurve
# from sage.vari_tree_lib.lightcurve import Lightcurve
# from sage.vari_tree_lib.lightcurve import Lightcurve
# from sage.vari_tree_lib.pickle_data import lightly_unpickle

grandparentDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # find our config file
config = configparser.ConfigParser()
config.read(os.path.join(grandparentDir, "tree_config.txt"))
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
data_dir = config["data_dir"]
pickled_df_dir = config["pickle_df_dir"]
twed_matrix = config["twed_matrix"]

def dlp(A, B):
    return np.abs(A - B)

def twed(lc_1, lc_2, nu=0.5, _lambda=1):
    # mags_1: magnitudes for lightcurve 1
    # times_1: timestamps for lightcurve 1
    # mags_2: magnitudes for lightcurve 2
    # times_2: timestamps for lightcurve 2
    # nu: elasticity parameter (>= 0 for distance measure)
    # _lambda: cost of deletion

    # zeroing both curves
    lc_1.times = lc_1.times - lc_1.times[0]
    lc_2.times = lc_2.times - lc_2.times[0]

    # subsequences of times where both curves have data
    end_time = min(lc_1.times[-1], lc_2.times[-1])
    lc_1 = Lightcurve.get_subseq(lc_1, 0, end_time)
    lc_2 = Lightcurve.get_subseq(lc_2, 0, end_time)


    mags_1 = lc_1.mags
    times_1 = lc_1.times
    mags_2 = lc_2.mags
    times_2 = lc_2.times

    # make sure data and timestamps have same length
    if len(mags_1) != len(times_1) or len(mags_2) != len(times_2):
        return None

    if nu < 0:
        return None

    # check for empty lightcurves
    if len(mags_1) == 0 or len(mags_2) == 0:
        return None
    
    if lc_1 == lc_2:
        return 0.0

    # reindex
    mags_1 = np.array([0] + list(mags_1))
    times_1 = np.array([0] + list(times_1))
    mags_2 = np.array([0] + list(mags_2))
    times_2 = np.array([0] + list(times_2))

    # mags_1 = mags_1[0::2]
    # times_1 = times_1[0::2]
    # mags_2 = mags_2[0::2]
    # times_2 = times_2[0::2]

    n = len(mags_1)
    m = len(mags_2)

    DP = np.zeros((n, m))
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # filling in the matrix
    for i in range(1, n):
        for j in range(1, m):
            # cost of operations list
            C = np.ones((3, 1)) * np.inf

            # deletion in lightcurve 1
            C[0] = (
                DP[i-1, j]
                + dlp(mags_1[i], mags_1[i-1])
                + nu*(times_1[i] - times_1[i-1])
                + _lambda
            )

            # deletion in lightcurve 2
            C[1] = (
                DP[i, j-1]
                + dlp(mags_2[j], mags_2[j-1])
                + nu*(times_2[j] - times_2[j-1])
                + _lambda
            )

            # keep both datapoints
            C[2] = (
                DP[i-1, j-1]
                + dlp(mags_1[i], mags_2[j])
                + dlp(mags_1[i-1], mags_2[j-1])
                + nu * (abs(times_1[i] - times_2[j]) + abs(times_1[i - 1] - times_2[j - 1]))
            )

            DP[i, j] = np.min(C)
    
    twed = DP[n - 1, m - 1]
    
    return twed


if __name__ == "__main__":
    # curve1 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J000000.19+320847.2.df.pkl")
    # curve2 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J000000.64+620043.9.df.pkl")
    # curve3 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J202331.36+284217.6.df.pkl")
    # curve4 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J184314.54-173311.1.df.pkl")

    # curve1.standardize()
    # curve2.standardize()
    # curve3.standardize()
    # curve4.standardize()

    # twed_mat = np.zeros((4, 4))

    # t0 = time.time()
    # twed_mat[0, 1] = twed(curve1, curve2)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[1, 0] = twed(curve1, curve2)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[0, 2] = twed(curve1, curve3)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[2, 0] = twed(curve1, curve3)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[0, 3] = twed(curve1, curve4)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[3, 0] = twed(curve1, curve4)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[1, 2] = twed(curve2, curve3)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[2, 1] = twed(curve2, curve3)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[1, 3] = twed(curve2, curve4)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[3, 1] = twed(curve2, curve4)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[2, 3] = twed(curve3, curve4)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # twed_mat[3, 2] = twed(curve3, curve4)
    # t1 = time.time()
    # print (t1-t0)
    # t0 = time.time()
    # print (twed_mat)

##############################################################################################################################
    
    directory = data_dir + "/g_band_lcs"
    count = 1
    names = {}

    total = 3 # 358862
    twed_mat = np.zeros((total, total))

    for filename in os.listdir(directory)[:2]:
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            names[filename[-26:-7]] = count
            count += 1
            if count == 4: break
            print (f, count)

        for name in names.keys():
            curve1 = ASASSN_Lightcurve.from_dat_file(f)
            curve2 = ASASSN_Lightcurve.from_dat_file(pickled_df_dir + "/ASASSN-V_" + name + ".df.pkl")

            if curve1 != curve2:
                twed_calc = twed(curve1, curve2)

                twed_mat[count-1, names[name]] = twed_calc
                twed_mat[names[name], count-1] = twed_calc
                
                print (count)
    
    # DF = pd.DataFrame(twed_mat)

    # DF.to_csv(twed_matrix)

'''
import numpy as np
import pandas as pd
import os
import sys
import configparser

sys.path.append('../sage/')
from vari_tree_lib.lightcurve import Lightcurve
from vari_tree_lib.lightcurve import ASASSN_Lightcurve
from vari_tree_lib.pickle_data import lightly_unpickle

grandparentDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # find our config file
config = configparser.ConfigParser()
config.read(os.path.join(grandparentDir, "tree_config.txt"))
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
data_dir = config["data_dir"]
pickled_df_dir = config["pickle_df_dir"]
twed_matrix = config["twed_matrix"]

def dlp(A, B, p=2):
    return np.abs(A - B)

def twed(lc_1, lc_2, nu=0.5, _lambda=1):
    # mags_1: magnitudes for lightcurve 1
    # times_1: timestamps for lightcurve 1
    # mags_2: magnitudes for lightcurve 2
    # times_2: timestamps for lightcurve 2
    # nu: elasticity parameter (>= 0 for distance measure)
    # _lambda: cost of deletion

    mags_1 = lc_1.mags
    times_1 = lc_1.times
    mags_2 = lc_2.mags
    times_2 = lc_2.times

    # make sure data and timestamps have same length
    if len(mags_1) != len(times_1) or len(mags_2) != len(times_2):
        return None

    if nu < 0:
        return None

    # check for empty lightcurves
    if len(mags_1) == 0 or len(mags_2) == 0:
        return None
    
    if lc_1 == lc_2:
        return 0.0

    # reindex
    mags_1 = np.array([0] + list(mags_1))
    times_1 = np.array([0] + list(times_1))
    mags_2 = np.array([0] + list(mags_2))
    times_2 = np.array([0] + list(times_2))

    n = len(mags_1)
    m = len(mags_2)

    DP = np.zeros((n, m))
    DP[0, :] = np.inf
    DP[:, 0] = np.inf
    DP[0, 0] = 0

    # filling in the matrix
    for i in range(1, n):
        for j in range(1, m):
            # cost of operations list
            C = np.ones((3, 1)) * np.inf

            # deletion in lightcurve 1
            C[0] = (
                DP[i-1, j]
                + dlp(mags_1[i], mags_1[i-1])
                + nu*(times_1[i] - times_1[i-1])
                + _lambda
            )

            # deletion in lightcurve 2
            C[1] = (
                DP[i, j-1]
                + dlp(mags_2[j], mags_2[j-1])
                + nu*(times_2[j] - times_2[j-1])
                + _lambda
            )

            # keep both datapoints
            C[2] = (
                DP[i-1, j-1]
                + dlp(mags_1[i], mags_2[j])
                + dlp(mags_1[i-1], mags_2[j-1])
                + nu * (abs(times_1[i] - times_2[j]) + abs(times_1[i - 1] - times_2[j - 1]))
            )

            DP[i, j] = np.min(C)
    
    twed = DP[n - 1, m - 1]
    
    return twed


if __name__ == "__main__":
    curve1 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J000000.19+320847.2.df.pkl")
    curve2 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J000000.64+620043.9.df.pkl")
    curve3 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J202331.36+284217.6.df.pkl")
    curve4 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J184314.54-173311.1.df.pkl")

    twed_mat = np.zeros((4, 4))
'''
