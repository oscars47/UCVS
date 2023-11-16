import pandas as pd
import sys, os, pickle
import bz2
from os.path import join, isfile

from tqdm import tqdm


def lightly_pickle(data, filename, protocol=4):
    f = open(filename + '.pkl', 'wb')
    pickle.dump(data, f, protocol=protocol)
    f.close()


def lightly_unpickle(filename):
    f = open(filename, 'rb')
    data = pickle.load(f)
    f.close()
    return data


def compress_and_pickle(data, filename, protocol=4):
    with bz2.BZ2File(filename + ".pbz2", "w") as f:
        pickle.dump(data, f, protocol=protocol)


def uncompress_pickle(filename):
    assert filename.endswith("pbz2")
    f = bz2.BZ2File(filename, "rb")
    data = pickle.load(f)
    return data


# batch pickle dataframe representations of unpickled, uncompressed files
# terrible programming practice: this function also silently pares down the data to just HJD, mag, and mag_err
def lightly_pickle_raw_data_files(filepaths, outdir):
    if not os.path.isdir(outdir):
        os.mkdir(outdir)
    for i in tqdm(range(len(filepaths)), desc='Pickling data', position=0, leave=True):
        path = filepaths[i]
        f = path.split(os.sep)[-1]
        df = pd.read_csv(path, sep='\t', usecols=["HJD", "mag", "mag_err","ML_classification"])
        lightly_pickle(df, join(outdir, f.replace(".dat", ".df")))


if __name__ == "__main__":
    import configparser

    config = configparser.ConfigParser()
    config.read("../tree_config.txt")
    config = config["DEFAULT"]
    # specify the path to the "g_band" folder of the UCVS data in your config file
    data_dir = config["data_dir"]
    sample_dir = config["sample_storage_dir"]
    lc_dir = os.path.join(data_dir, "g_band_lcs")
    pickled_df_dir = config["pickle_df_dir"]

    with open("all_lightcurve_names.txt", "r") as f:
        filepaths = [os.path.join(lc_dir, p.replace("\n", '')) for p in f.readlines()]

    files = filepaths[20000:]
    lightly_pickle_raw_data_files(files, pickled_df_dir)

