import os, sys
import math
import random
import pandas as pd
import configparser
import json

config = configparser.ConfigParser()
config.read("../tree_config.txt")
config = config["DEFAULT"]
data_dir = config["data_dir"]

filepath = os.join(data_dir, 'asassn_rounded.csv')

def sample_num_curves(path, num_samples, sample_method):
    #iterate through summary csv -- compile dict of curves by class
    if not os.path.exists('lightcurve_names_by_class.json'):
        d = {}
        df = pd.read_csv(path)
        df = pd.DataFrame(df, columns = ['ID', 'ML_classification'])
        for row in df.index:
            if df['ML_classification'][row] in d:
                d[df['ML_classification'][row]].append(df['ID'][row])
            else:
                d[df['ML_classification'][row]] = [df['ID'][row]]
        with open('lightcurve_names_by_class.json', 'w') as output:
            json.dump(d, output)
    with open('lightcurve_names_by_class.json', 'r') as input:
        d = json.load(input)

    #figure out inverse number of curves to select (if applicable) based on user input
    if (sample_method == 'equal'):
        samples_by_class = {}
        for key in d:
            samples_by_class[key] = num_samples
    
    if (sample_method == 'inverse_freq'):
        num_samples = num_samples * len(list(d.keys()))
        key0 = list(d.keys())[0]
        inv_ratios = {}
        for key in d:
            inv_ratios[key] = len(d[key0]) / len(d[key])
        total = 0
        for key in inv_ratios:
            total += inv_ratios[key]
        c0_samples = num_samples / total
        samples_by_class = {}
        for key in d:
            samples_by_class[key] = math.ceil(c0_samples * inv_ratios[key])

    #select, number of curves required of each class. 
    #store as txt file, new line between each curve
    samples = []
    
    for key in d:
        samples = samples + random.sample(d[key], samples_by_class[key])
        
    for i in range(0, len(samples)):
        samples[i] = str(samples[i])
    
    with open('tree_samples.txt', 'w') as output:
        output.write('\n'.join(samples))

if __name__ == "__main__":
    dataset = "ASASSN"
    num_curves = 100
    sample_num_curves(filepath, 20, )
