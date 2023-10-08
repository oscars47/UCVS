# by @oscars47 and @ghirsch123

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from scipy.interpolate import griddata

from nnprep import *

# load model
MODEL_PATH = 'ENTER THE DIRECTORY THAT HAS THE LOCATION OF THE MODEL'
model = load_model(os.path.join(MODEL_PATH, 'model-best.h5'))

# load input_x and output targets
DATA_DIR = 'ENTER THE DIRECTORY THAT HAS ALL OF YOUR DATA IN IT'


input_x = np.load(os.path.join(DATA_DIR, 'mm_n_extra.npy'))
output_targets = np.load(os.path.join(DATA_DIR, 'targets_extra.npy'))


#load asasn-sn variables in last 50%
asassn = pd.read_csv(os.path.join(DATA_DIR, 'folded_mm_per.csv'))
min_index = int(0.5*len(asassn))
asassn = asassn.iloc[min_index:, :]

# takes in input probability vector
def convert_to_class(vec):
    # find index of largest arg
    index = np.argmax(vec)
    confidence = np.max(vec)
    class_type = int_to_class[index]
    return index, class_type, confidence

# get class names
object_names = asassn['id'].to_list()

# takes in input data which is array of arrays
def predict_vars(model, names, input_x, output_targets, file_name):
    results = model.predict(input_x) # predict!!
    # initialize lists to hold results
    output_classes = []
    output_classes_indices = []
    output_confidences= []
    for result in results:
        index, class_type, confidence = convert_to_class(result)
        output_classes_indices.append(index)
        output_classes.append(class_type)
        output_confidences.append(confidence)

    #convert targets back to classes
    output_target_names = []
    output_target_indices = []
    for target in output_targets:
        index, class_type, _ = convert_to_class(target)
        output_target_indices.append(index)
        output_target_names.append(class_type)

    # build and save dataset
    asassn_pred = pd.DataFrame()
    asassn_pred['ID'] = names
    asassn_pred['prediction'] = output_classes
    asassn_pred['confidence'] = output_confidences
    asassn_pred['actual'] = output_target_names
    
    asassn_pred.to_csv(os.path.join(DATA_DIR, file_name))
    return output_classes_indices, output_target_indices

def get_3dconfusion_matrix(output_targets, output_preds):
    # create a confusion matrix to illustrate results
    cm = confusion_matrix(output_targets, output_preds)
    cm_df = pd.DataFrame(cm, index = unique_targets, columns = unique_targets)
    # compute accuracy
    accuracy = 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if i ==j:
                accuracy += cm[i][j]
    accuracy /= len(output_preds) # divide total correct by total obs
    cm_norm_df = cm_df / cm_df.sum()
    cm_norm_matrix = np.array(cm_norm_df) # divide each column by the sum for that column to determine relative precentage

    # 3dimensional confusion matrix
    tuple_ls = []
    for i in range(len(unique_targets)):
        for j in range(len(unique_targets)):
            tuple_ls.append((i, j, cm[i][j], cm_norm_matrix[i][j]))
    # unzip
    x, y, z, c = zip(*tuple_ls)
    x = np.array(x)
    y = np.array(y)
    z = np.array(z)
    c = np.array(c)

    xi = np.linspace(x.min(), x.max(), 100)
    yi = np.linspace(y.min(), y.max(), 100)

    X,Y = np.meshgrid(xi,yi)

    Z = griddata((x,y),z,(X,Y), method='cubic')
    C = griddata((x,y),c,(X,Y), method='cubic')


    fig = go.Figure(go.Surface(x=xi,y=yi,z=Z, surfacecolor=C, colorscale='viridis'))
    fig.update_layout(title='Confusion matrix, accuracy = %f'%np.round(accuracy, 4), autosize=False,
                    width=1000, height=1000)
    fig.show()

def get_2dconfusion_matrix(output_targets, output_preds):
     # create a confusion matrix to illustrate results
    cm = confusion_matrix(output_targets, output_preds)
    cm_df = pd.DataFrame(cm, index = unique_targets, columns = unique_targets)
    # compute accuracy
    accuracy = 0
    for i in range(len(cm)):
        for j in range(len(cm[i])):
            if i ==j:
                accuracy += cm[i][j]
    accuracy /= len(output_preds) # divide total correct by total obs
    cm_norm_df = cm_df / cm_df.sum()
    cm_norm_matrix = np.array(cm_norm_df) # divide each column by the sum for that column to determine relative precentage
    # plot 2d confusion matrix
    plt.figure(figsize=(10,7))
    sns.heatmap(cm_norm_df, cmap = 'viridis', annot=True)
    plt.title('Confusion matrix, accuracy = %f'%np.round(accuracy, 4), fontsize=20)
    plt.ylabel('Actual variable class', fontsize=16)
    plt.xlabel('Predicted variable class', fontsize=16)
    #plt.savefig(os.path.join(DATA_DIR, 'confusion_acc_v0.0.1.jpeg'))
    plt.show()

file_name = 'best_model_results.csv'
output_classes_indices, output_target_indices = predict_vars(model, object_names, input_x, output_targets, file_name)
get_2dconfusion_matrix(output_classes_indices, output_target_indices)
get_3dconfusion_matrix(output_classes_indices, output_target_indices)


