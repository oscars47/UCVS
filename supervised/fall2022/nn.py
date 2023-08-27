# file to load the nn

import pandas as pd
import numpy as np
import os
import tensorflow as tf
from keras import layers
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
import wandb
from wandb.keras import WandbCallback

# read in our data
DATA_DIR = '/home/oscar47/Desktop/astro101/data/g_band/var_output/v0.1.1'

# check if keras recognizes gpu
print("Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')))

train_x_ds = np.load(os.path.join(DATA_DIR, 'train_x_ds.npy'))
val_x_ds = np.load(os.path.join(DATA_DIR, 'val_x_ds.npy'))
train_y_ds = np.load(os.path.join(DATA_DIR, 'train_y_ds.npy'))
val_y_ds = np.load(os.path.join(DATA_DIR, 'val_y_ds.npy'))

input_shape = train_x_ds[0].shape
output_len = len(train_y_ds[0])

# build model functions--------------------------------
def build_model(size1, size2, size3, size4, size5, dropout, learning_rate):
    model = Sequential()

    model.add(layers.Dense(size1))
    model.add(layers.Dense(size2))
    model.add(layers.Dense(size3))
    model.add(layers.Dense(size4))
    model.add(layers.Dense(size5))

    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_len))

    # return len of class size
    model.add(layers.Dense(output_len))
    model.add(layers.Activation('softmax'))

    optimizer = Adam(learning_rate = learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

# build model functions--------------------------------
def build_model_small(size1, dropout):
    model = Sequential()

    model.add(layers.Dense(size1))
    model.add(layers.Dense(size1))
    model.add(layers.Dense(size1))
    
 
    model.add(layers.Dropout(dropout))
    model.add(layers.Dense(output_len))

    # return len of class size
    model.add(layers.Dense(output_len))
    model.add(layers.Activation('softmax'))

    optimizer = Adam()
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model


def train(config=None):
    with wandb.init(config=config):
    # If called by wandb.agent, as below,
    # this config will be set by Sweep Controller
      config = wandb.config

      #pprint.pprint(config)

      #initialize the neural net; 
      global model
      model = build_model(config.size_1,  config.size_2, config.size_3, 
              config.size_4, config.size_5, 
              config.dropout, config.learning_rate)
      
      #now run training
      history = model.fit(
        train_x_ds, train_y_ds,
        batch_size = config.batch_size,
        validation_data=(val_x_ds, val_y_ds),
        epochs=config.epochs,
        callbacks=[WandbCallback()] #use callbacks to have w&b log stats; will automatically save best model                     
      )

def train_manual():
    global model
    model = build_model_small(256, 0.2)
    
    #now run training
    history = model.fit(
    train_x_ds, train_y_ds,
    batch_size = 128,
    validation_data=(val_x_ds, val_y_ds),
    epochs=10
    )

# set dictionary with random search; optimizing val_loss--------------------------
sweep_config= {
    'method': 'random',
    'name': 'val_accuracy',
    'goal': 'maximize'
}

sweep_config['metric']= 'val_accuracy'

# now name hyperparameters with nested dictionary
# parameters_dict = {
#     'epochs': {
#        'distribution': 'int_uniform',
#        'min': 10,
#        'max': 20
#     },
#     # for build_dataset
#      'batch_size': {
#        'distribution': 'q_log_uniform',  #we want to specify a distribution type to more efficiently iterate through these hyperparams
#        'q': 8,
#        'min': np.log(64),
#        'max': np.log(256)
#     },
#     'size_1': {
#        'distribution': 'q_log_uniform',
#        'q': 8,
#        'min': np.log(64),
#        'max': np.log(256)
#     },
#     'size_2': {
#        'distribution': 'q_log_uniform',
#        'q': 8,
#        'min': np.log(64),
#        'max': np.log(256)
#     },
#      'size_3': {
#        'distribution': 'q_log_uniform',
#        'q': 8,
#        'min': np.log(64),
#        'max': np.log(256)
#     },
#      'size_4': {
#        'distribution': 'q_log_uniform',
#        'q': 8,
#        'min': np.log(64),
#        'max': np.log(256)
#     },
#      'size_5': {
#        'distribution': 'q_log_uniform',
#        'q': 8,
#        'min': np.log(64),
#        'max': np.log(256)
#     },
#     'dropout': {
#       'distribution': 'uniform',
#        'min': 0,
#        'max': 0.6
#     },
#     'learning_rate':{
#          #uniform distribution between 0 and 1
#          'distribution': 'uniform', 
#          'min': 0,
#          'max': 0.1
#      }
# }

parameters_dict = {
    'epochs': {
       'distribution': 'int_uniform',
       'min': 20,
       'max': 100
    },
    # for build_dataset
     'batch_size': {
       'values': [x for x in range(32, 161, 32)]
    },
    'size_1': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'size_2': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_3': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_4': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },'size_5': {
       'distribution': 'int_uniform',
       'min': 64,
       'max': 256
    },
    'dropout': {
      'distribution': 'uniform',
       'min': 0,
       'max': 0.6
    },
    'learning_rate':{
         #uniform distribution between 0 and 1
         'distribution': 'uniform', 
         'min': 0,
         'max': 0.1
     }
}

# append parameters to sweep config
sweep_config['parameters'] = parameters_dict 

# login to wandb----------------
wandb.init(project="Astro101_Project_NewData2", entity="oscarscholin")

# initialize sweep agent
sweep_id = wandb.sweep(sweep_config, project='Astro101_Project_NewData2', entity="oscarscholin")
wandb.agent(sweep_id, train, count=100)

#train_manual()