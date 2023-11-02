import math
import os
import random
import sys
import timeit

from line_profiler_pycharm import profile
import matplotlib.colors as mcolors
from tqdm import tqdm
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import astropy
from vari_tree_lib.pickle_data import lightly_unpickle
import configparser
# read in configuration file to set certain parameters. these files should NOT be tracked by github!
# each user should copy tree_config_template.txt and rename it to tree_config.txt
# then, they should add the path of tree_config.txt to their .gitignore file, creating one if they don't have it already
#
# IMPORTANT: when you add settings to the code and the config, you should also add them to the tree_config_template.txt file!!
#       Include a comment that describes what your setting means, but also give it a descriptive name
# if you pull code and you start getting a key error on a config item, check that your tree_config.txt is up to date with tree_config_template.txt
grandparentDir = os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir))  # find our config file
config = configparser.ConfigParser()
config.read(os.path.join(grandparentDir, "tree_config.txt"))
config = config["DEFAULT"]
# specify the path to the "g_band" folder of the UCVS data in your config file
data_dir = config["data_dir"]
pickled_df_dir = config["pickle_df_dir"]

lc_dir = os.path.join(data_dir, "g_band_lcs")

colors = list(mcolors.XKCD_COLORS)


class Lightcurve:
    @profile
    def __init__(self, time_arr, mag_arr, mag_err_arr, ID, obj_type=None, parent_id=None, standardized=False):
        """
        Representation of a lightcurve (time-magnitude series)

        @param ID: The ID of the sequence. [ID]_n is the nth subsequence of curve [ID]
        @param obj_type: classification of lightcurve, if known
        @param parent_id: if this curve is a subsequence, this is the id of the curve it was sampled from
        @type time_arr: np.array
        @type mag_arr: np.array
        @type mag_err_arr: np.array
        @type ID: str
        """
        self.times = time_arr
        self.mags = mag_arr
        self.mag_err = mag_err_arr
        self.id = ID
        self.obj_type = obj_type
        self.parent_id = parent_id
        self.num_subseqs = 0
        self._marker_color = None  # for plotting
        self.standardized = standardized

    @property
    def marker_color(self):
        # don't spend time choosing unless we're actually going to plot this curve
        if self._marker_color is None:
            self._marker_color = random.choice(colors)
        return self._marker_color

    @property
    def total_time(self):
        if len(self.times):
            return self.times[-1] - self.times[0]
        return 0

    @profile
    def standardize(self, verbose=False):
        """
        Standardize this lightcurve inplace: subtract mean mag and divide by std dev to get mean mag of 0 and std dev of 1
        @rtype: None
        """
        if self.standardized:
            print(f"Warning! Standardizing curve {self.id} (already standardized)")
        mean_mag = np.mean(self.mags)
        std_dev_mag = np.std(self.mags)
        if verbose:
            print("Before standardization:")
            print(f"Mean mag: {mean_mag}")
            print(f"Standard deviation: {std_dev_mag}")
        self.mags -= mean_mag
        self.mags /= std_dev_mag
        self.standardized = True
        if verbose:
            mean_mag = np.mean(self.mags)
            std_dev_mag = np.std(self.mags)
            print("After standardization:")
            print(f"Mean mag: {mean_mag}")
            print(f"Standard deviation: {std_dev_mag}")

    def _make_plot(self, coplot=None, fig=None, ax=None, alpha=1):
        if not ax:
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.set_xlabel('Time (HJD)', fontsize=14)
            ax.set_ylabel('Mag', fontsize=14)
            ax.set_title('Lightcurve for Object ' + self.id, fontsize=16)
        else:
            assert fig
        ax.scatter(self.times, self.mags, alpha=alpha)
        # plt.errorbar(self.times, self.mags, yerr=self.mag_err, fmt='o', color=random.choice(colors))
        if coplot:
            for lightcurve in coplot:
                ax.scatter(lightcurve.times, lightcurve.mags, c=lightcurve.marker_color)
        return fig

    def plot(self, coplot=None):
        fig = self._make_plot(coplot=coplot)
        fig.show()

    # not in use, for posterity
    @profile
    def manual_get_subseq(self, start_time, end_time):
        # bounds checking
        if self.times[0] > end_time or self.times[-1] < start_time:
            self.num_subseqs += 1
            return Lightcurve([], [], [], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id, obj_type=self.obj_type, standardized=False)

        # find first and last time indices
        curve_start = None
        curve_end = None
        for i in range(len(self.times)):
            if curve_start is None and self.times[i] >= start_time:
                curve_start = i
            if self.times[i] > end_time:
                curve_end = i
                break
        if curve_end is None:
            # we ran out of sequence before we hit the end - that's ok, we just need to set our end time or it will be None
            curve_end = len(self.times)

        sub_times = self.times[curve_start:curve_end]
        sub_mags = self.mags[curve_start:curve_end]
        sub_mag_err = self.mag_err[curve_start:curve_end]

        # print(f"making subseq for window between {start_time} and {end_time} with {len(sub_times)} data points.")
        self.num_subseqs += 1
        return Lightcurve(sub_times, sub_mags, sub_mag_err, ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id,
                          obj_type=self.obj_type, standardized=self.standardized)

        # gather times in between
        # associate with magnitude and mag errr
        # construct and return lightcurve object

    @profile
    def get_subseq(self, start_time, end_time):
        if self.times[0] > end_time or self.times[-1] < start_time:
            self.num_subseqs += 1
            return Lightcurve([], [], [], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id, obj_type=self.obj_type)
        curve_indices = np.where(np.logical_and(start_time <= self.times, self.times<end_time))[0]
        self.num_subseqs += 1
        return Lightcurve(self.times[curve_indices], self.mags[curve_indices], self.mag_err[curve_indices], ID=f"{self.id}_{self.num_subseqs}", parent_id=self.id,
                          obj_type=self.obj_type)

    @profile
    def random_subsequence(self, window_duration):
        if not self.standardized:
            self.standardize()
        index_choices = np.where(self.times < self.times[-1]-window_duration)[0][-1]
        start_idx = random.choice(range(index_choices))
        start = self.times[start_idx]
        end = start + window_duration
        return self.get_subseq(start, end)

    @profile
    def sliding_subseqs(self, window_duration, time_interval):
        """
        Sample a window of duration window_duration, then move ahead time_interval (time_interval<<window_duration) and repeat
        @param window_duration: duration of subsequence, days
        @param time_interval: days between beginnings of each subsequence
        @return: list of subseqs
        """
        i = 0
        start = self.times[0]
        end = start + window_duration
        subseqs = []
        while start + i < self.times[-1]:
            subseqs.append(self.get_subseq(start + i, end + i))
            i += time_interval
        return subseqs


    def plot_subseqs(self, subsequences, start_times, end_times, also_plot_individual=False):
        """
        Plot this lightcurve, then overplot the curves of subsequences provided.
        @param subsequences: Lightcurve objects
        @param start_times: start times of the subsequences
        @param end_times: start times of the subsequences
        @param also_plot_individual: bool. if True, will also plot each subseq on its own axis in the figure
        """
        assert len(subsequences) == len(start_times) and len(start_times) == len(end_times)
        if also_plot_individual:
            fig, axes = plt.subplots(len(subsequences)+1, sharex=True, sharey=True, figsize=(18, 10))
            ax = axes[0]
            ax.set_title(f"Lightcurve", fontsize="x-small", loc="left", verticalalignment="top")
        else:
            fig, ax = plt.subplots()

        fig = self._make_plot(coplot=subsequences, fig=fig, ax=ax)
        i = 0
        for subseq, start, end in zip(subsequences, start_times, end_times):
            # draw lines showing where it should have sliced the subseq
            ax.axvline(x=start, color=subseq.marker_color, linestyle='--',
                       label=f't={start}')
            ax.axvline(x=end, color=subseq.marker_color, linestyle='--',
                       label=f't={end}')
            if also_plot_individual:
                axes[i+1].scatter(subseq.times, subseq.mags, c=subseq.marker_color)
                axes[i+1].set_title(f"Subsequence {i+1}",fontsize="x-small",loc="left",verticalalignment="top")
            i += 1
        fig.supxlabel('Time (HJD)', fontsize=25 if also_plot_individual else "large")
        fig.supylabel('Mag', fontsize=25 if also_plot_individual else "large",horizontalalignment="left")
        fig.suptitle('Lightcurve for Object ' + self.id, fontsize=25 if also_plot_individual else "large")
        fig.legend()
        fig.show()


# subclass of the general lightcurve class that represents an ASAS-SN lightcurve read from a .dat file
class ASASSN_Lightcurve(Lightcurve):
    @classmethod
    @profile
    def from_dat_file(cls, path: str):
        """
        Read a provided .dat file representing an ASAS-SN lightcurve.
        @rtype: ASASSN_Lightcurve
        """
        lc_df = pd.read_csv(path, sep='\t',usecols=["HJD", "mag", "mag_err"])
        time = lc_df['HJD'].astype(np.float32).to_numpy()  # convert the times to floats
        mag = np.array([float(v.replace(">", '').replace("<", '')) for v in
               lc_df['mag'].astype(str)])  # convert the mags, remove "<" and ">"
        mag_err = lc_df['mag_err'].astype(np.float32).to_numpy()
        obj_type = lc_df["ML_classification"]
        ID = ASASSN_Lightcurve.id_from_filename(path)
        return cls(time, mag, mag_err, ID=ID,obj_type=obj_type)

    @classmethod
    @profile
    def from_pickle(cls, path: str):
        """
        Read a provided .dat file representing an ASAS-SN lightcurve.
        @rtype: ASASSN_Lightcurve
        """
        lc_df = lightly_unpickle(path)
        time = lc_df['HJD'].astype(np.float32).to_numpy()  # convert the times to floats
        mag = np.array([float(v.replace(">", '').replace("<", '')) for v in
               lc_df['mag'].astype(str)])  # convert the mags, remove "<" and ">"
        mag_err = lc_df['mag_err'].astype(np.float32).to_numpy()
        obj_type = lc_df["ML_classification"]
        ID = ASASSN_Lightcurve.id_from_filename(path)
        return cls(time, mag, mag_err, ID=ID,obj_type=obj_type)

    @staticmethod
    def id_from_filename(filename):
        ID = filename.split(os.sep)[-1].split('.dat')[0].split('_')  # parse the ID
        ID = ID[0] + '_' + ID[1]
        ID = ID.replace("-", "_").replace("+", "_").replace(".", "_")
        return ID


if __name__ == "__main__":
    # this main function is just used for testing stuff

    # "ASASSN-V_J055803.37-143014.8.dat"

    # this was used to create "lightcurves.txt" (a list of a couple lightcurve names so i don't have to call os.listdir):
    # files = os.listdir(pickled_df_dir)
    # with open("../all_pickled_lightcurve_names.txt", "w+") as f:
    #     f.write('\n'.join(files))
    # exit()

    with open("../all_pickled_lightcurve_names.txt", "r") as f:
        files = [p.replace("\n", '') for p in f.readlines()]

    # files = files[:20]
    # for i in tqdm(range(len(files)), desc='plotting lightcurves', position=0, leave=True):
    #     f = files[i]
    #     seq = ASASSN_Lightcurve.from_dat_file(os.path.join(lc_dir, f))
    seq = ASASSN_Lightcurve.from_pickle(os.path.join(pickled_df_dir, random.choice(files)))
        # seq.plot()
    start = seq.times[math.floor(len(seq.times) / 4)]
    end = seq.times[math.floor(3 * len(seq.times) / 4)]

    sub_seq = seq.get_subseq(start, end)
    # seq.plot(coplot=[sub_seq])
    # seq.plot()
    # seq.plot_subseqs(subsequences=[sub_seq],start_times=[start],end_times=[end])

    subseqs = []
    starts = []
    ends = []
    timewindow = 250  # days
    offset = 200  # days
    start = seq.times[0]
    end = start+timewindow
    i = 0
    num_repeats = 3
    # print(f"Manual: {timeit.timeit(lambda: seq.manual_get_subseq(start+i, end+i), number=num_repeats)}s")
    print(f"Random subseq: {timeit.timeit(lambda: seq.random_subsequence(i), number=num_repeats)}s")

    while start+i < seq.times[-1]-timewindow:
        subseqs.append(seq.get_subseq(start+i, end+i))
        starts.append(start+i)
        ends.append(end+i)
        i += offset

    seq.plot_subseqs(subsequences=subseqs, start_times=starts, end_times=ends, also_plot_individual=True)

    subseqs = seq.sliding_subseqs(250,10)
    print(subseqs)