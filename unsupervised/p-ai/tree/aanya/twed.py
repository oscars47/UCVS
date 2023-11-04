import numpy as np
import pandas as pd
import os
import sys

sys.path.append('../sage/')
from vari_tree_lib.lightcurve import Lightcurve
from vari_tree_lib.lightcurve import ASASSN_Lightcurve
from vari_tree_lib.pickle_data import lightly_unpickle


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
    # curve1 = ASASSN_Lightcurve.from_pickle("/Users/aanyapratapneni/Downloads/P-ai Data/pickle_df/ASASSN-V_J000000.19+320847.2.df.pkl")
    # curve2 = ASASSN_Lightcurve.from_pickle("/Users/aanyapratapneni/Downloads/P-ai Data/pickle_df/ASASSN-V_J000000.64+620043.9.df.pkl")
    # print (twed(curve1, curve2))
    
    directory = '/Users/aanyapratapneni/Downloads/P-ai Data/pickle_df/'
    count = 1
    names = {}

    total = 358862
    twed_matrix = np.zeros((total, total))

    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        if os.path.isfile(f):
            names[filename[-26:-7]] = count
            count += 1
            print (count)
    
        for name in names.keys():
            curve1 = ASASSN_Lightcurve.from_pickle(f)
            curve2 = ASASSN_Lightcurve.from_pickle("/Users/aanyapratapneni/Downloads/P-ai Data/pickle_df/ASASSN-V_" + name + ".df.pkl")
            twed_calc = twed(curve1, curve2)

            twed_matrix[count-1, names[name]] = twed_calc
            twed_matrix[names[name], count-1] = twed_calc
    
    DF = pd.DataFrame(twed_matrix)
    DF.to_csv("/Users/aanyapratapneni/Downloads/data1.csv")