# file to predict based on trained neural network

import os
import numpy as np
import pandas as pd
from keras.models import load_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from nn_prep import int_to_class


def predict_vars(model, names, input_x, output_targets, file_name, model_dir):
    '''Predicts the class of the input data and saves the results to a csv file.

    Params:
        model (keras model): trained neural network
        names (list): list of the names of the objects
        input_x (numpy array): numpy array of the input data
        output_targets (numpy array): numpy array of the output targets
        file_name (str): name of the file to save the results to
        model_dir (str): path to the models directory

    Returns:
        None, but saves the results to a csv file:
            ID (str): name of the object
            prediction (str): predicted class type
            prediction_index (int): index of the predicted class type
            confidence (float): confidence of the prediction (the probability of the class type)
            actual (str): actual class type
            actual_index (int): index of the actual class type
    
    '''
    def convert_to_class(vec):
        '''Converts the probability vector to a class type by finding instance of largest probability.

        Params:
            vec (numpy array): probability vector

        Returns:
            index (int): index of the class type
            class_type (str): class type
            confidence (float): confidence of the prediction (the probability of the class type)
        '''
        index = np.argmax(vec)
        confidence = np.max(vec)
        class_type = int_to_class[index]
        return index, class_type, confidence
    
    # use .predict() method to get probability vectors
    results = model.predict(input_x)

    # initialize lists to hold results
    output_classes = []
    output_classes_indices = []
    output_confidences= []
    for result in results:
        index, class_type, confidence = convert_to_class(result)
        output_classes_indices.append(index)
        output_classes.append(class_type)
        output_confidences.append(confidence)

    # convert targets back to classes
    output_target_names = []
    output_target_indices = []
    for target in output_targets:
        index, class_type, _ = convert_to_class(target)
        output_target_indices.append(index)
        output_target_names.append(class_type)

    # build and save dataset with indices so we can make 
    pred = pd.DataFrame()
    pred['ID'] = names
    pred['prediction'] = output_classes
    pred['prediction_index'] = output_classes_indices
    pred['confidence'] = output_confidences
    pred['actual'] = output_target_names
    pred['actual_index'] = output_target_indices
    pred.to_csv(os.path.join(model_dir, file_name))

def get_accuracy(pred):
    '''Computes the overall accuracy # correct / total # of observations.
    
    Params:
        pred (DataFrame): DataFrame of the predictions

    Returns:
        accuracy (float): overall accuracy
    
    '''

    # compute accuracy: dataframe has indices of each class, so we can just compare the two
    accuracy = 0
    for i in range(len(pred)):
        if pred['prediction_index'][i] == pred['actual_index'][i]:
            accuracy += 1
    accuracy /= len(pred)
    
    return accuracy

def get_confusion_matrix(pred, model_name, data_name):
    '''Computes the confusion matrix to visualize the predictions.

    Params:
        pred (DataFrame): DataFrame of the predictions
        model_name (str): name of the model
        data_name (str): name of the dataset (e.g., train, val, test)

    Returns:
        None, but saves the confusion matrix as a pdf file
    
    
    '''

    # create a confusion matrix to illustrate results
    unique_targets = pred['actual'].unique()
    # sort so that the order is the same as the confusion matrix
    unique_targets.sort()
    cm = confusion_matrix(pred['actual'], pred['prediction'])
    cm_df = pd.DataFrame(cm, index = unique_targets, columns = unique_targets)
    # compute accuracy
    accuracy = get_accuracy(pred)

    # divide each column by the sum for that column to determine relative precentage
    cm_norm_df = cm_df / cm_df.sum()
    cm_norm_matrix = np.array(cm_norm_df)

    # plot confusion matrix
    fig, ax = plt.subplots(figsize=(10,10))
    # add colorbar
    cax = ax.matshow(cm_norm_matrix, cmap=plt.cm.viridis, alpha=0.7)
    # make colorbar same size as actual plot
    fig.colorbar(cax, fraction=0.046, pad=0.04)
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title(f'Confusion Matrix, accuracy = {np.round(accuracy, 4)}')

    # set the ticks at the center of the cells
    ax.set_xticks(np.arange(len(unique_targets)))
    ax.set_yticks(np.arange(len(unique_targets)))
    
    # add labels to axes corresponding to the class name
    ax.set_xticklabels([''] + unique_targets)
    ax.set_yticklabels([''] + unique_targets)
    plt.tight_layout()
    plt.savefig(os.path.join(MODEL_DIR, f'confusion_matrix_{model_name}_{data_name}.pdf'))

if __name__ == '__main__':
    # define path to where your data is
    DATA_DIR = '/Users/oscarscholin/Desktop/Pomona/Senior_Year/Fall2024/Astro_proj/UCVS/data'
    MODEL_DIR = '/Users/oscarscholin/Desktop/Pomona/Senior_Year/Fall2024/Astro_proj/UCVS/supervised/fall2023/models'

    # load datasets #
    train_x_ds = np.load(os.path.join(DATA_DIR, 'train_x_ds.npy'))
    train_y_ds = np.load(os.path.join(DATA_DIR, 'train_y_ds.npy'))
    train_names = pd.read_csv(os.path.join(DATA_DIR, 'train_names.csv'))
    train_names = train_names['0'].to_list()

    val_x_ds = np.load(os.path.join(DATA_DIR, 'val_x_ds.npy'))
    val_y_ds = np.load(os.path.join(DATA_DIR, 'val_y_ds.npy'))
    val_names = pd.read_csv(os.path.join(DATA_DIR, 'val_names.csv'))
    val_names = val_names['0'].to_list()

    x_test = np.load(os.path.join(DATA_DIR, 'x_test.npy'))
    y_test = np.load(os.path.join(DATA_DIR, 'y_test.npy'))
    test_names = pd.read_csv(os.path.join(DATA_DIR, 'test_names.csv'))
    test_names = test_names['0'].to_list()

    # load model
    model_name = 'oscar_model1'
    model = load_model(os.path.join(MODEL_DIR, f'{model_name}.keras'))

    # predict on test set  - REMEMBER, THIS IS SACRED!!!
    predict_vars(model, test_names, x_test, y_test, f'test_pred_{model_name}.csv', MODEL_DIR)

    # predict on validation set
    predict_vars(model, val_names, val_x_ds, val_y_ds, f'val_pred_{model_name}.csv', MODEL_DIR)

    # predict on training set 
    predict_vars(model, train_names, train_x_ds, train_y_ds, f'train_pred_{model_name}.csv', MODEL_DIR)

    # compute accuracy
    test_pred = pd.read_csv(os.path.join(MODEL_DIR, f'test_pred_{model_name}.csv'))
    val_pred = pd.read_csv(os.path.join(MODEL_DIR, f'val_pred_{model_name}.csv'))
    train_pred = pd.read_csv(os.path.join(MODEL_DIR, f'train_pred_{model_name}.csv'))


    # get confusion matrix
    get_confusion_matrix(val_pred, model_name, 'val')
    get_confusion_matrix(train_pred, model_name, 'train')
    get_confusion_matrix(test_pred, model_name, 'test')