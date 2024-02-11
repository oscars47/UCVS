import numpy as np
import pandas as pd
import os
import sys
import configparser
import time
from random import randint
from multiprocessing import Pool

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir,"sage")))  # find our config file)
from vari_tree_lib.lightcurve import ASASSN_Lightcurve
from vari_tree_lib.lightcurve import Lightcurve
from vari_tree_lib.pickle_data import lightly_unpickle

grandparentDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir))  # find our config file
config = configparser.ConfigParser()
config.read(os.path.join(grandparentDir, "tree_config.txt"))
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
data_dir = config["data_dir"]
pickled_df_dir = config["pickle_df_dir"]
matrix_path = config["twed_matrix"]
subseq_dir = config["subseq_dir"]

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

def do_twed(i,j,f,f2):
    curve1 = lightly_unpickle(f)
    curve2 = lightly_unpickle(f2)

    twed_calc = twed(curve1, curve2)

    return i,j, twed_calc


if __name__ == "__main__":
    names = {}

    for num in [20,50,100,200,500]:
        st = time.perf_counter()
        num_subseqs = num # 358862
        twed_mat = np.zeros((num_subseqs, num_subseqs))

        files = [os.path.join(subseq_dir,f) for f in os.listdir(subseq_dir)[:num_subseqs]]
        params = []

        for i, f in enumerate(files):
            # names[ASASSN_Lightcurve.id_from_filename(f)] = count

            for j, f2 in enumerate(files):
                if i <= j:
                    continue
                params.append((i, j, f, f2))
                        
        # DF = pd.DataFrame(twed_mat)

        # DF.to_csv(matrix_path)
        # print(twed_mat)

        with Pool() as pool:
            res = pool.starmap(do_twed, params)

        for item in res:
            twed_mat[item[0], item[1]] = item[2]
            twed_mat[item[1], item[0]] = item[2]
        
        d = time.perf_counter() - st

        print (num, d)