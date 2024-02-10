import numpy as np
import pandas as pd
import os
import sys
import configparser
import time

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

if __name__ == "__main__":
    curve1 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J000000.19+320847.2.df.pkl")
    curve2 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J000000.64+620043.9.df.pkl")
    curve3 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J202331.36+284217.6.df.pkl")
    curve4 = ASASSN_Lightcurve.from_pickle(pickled_df_dir + "/ASASSN-V_J184314.54-173311.1.df.pkl")

curve1.times = curve1.times - curve1.times[0]
print (curve1.times)

# print ((curve1.times - curve1.times[0])[-1])
# print ((curve4.times - curve4.times[0])[-1])