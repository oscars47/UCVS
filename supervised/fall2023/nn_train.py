# file to train the neural network using Keras

import os
import numpy as np
from keras import layers
from keras.models import Sequential
from keras.optimizers import Adam
import matplotlib.pyplot as plt

## define neural net architecture ##
def build_model(size1, size2, size3, dropout, learning_rate, output_len):
    '''Sample 3 hidden layer NN with dropout parameter. Used inside train() function.
    
    Params:
        size1 (int): size of first hidden layer (# of neurons)
        size2 (int): size of second hidden layer
        size3 (int): size of third hidden layer
        dropout (float): dropout parameter (randomly drops certain neurons to prevent overfitting and speed up training)
        learning_rate (float): learning rate of the optimizer; how aggressively the optimizer will try to minimize the loss function
        output_len (int): length of output vector

    Returns:
        model (Sequential): keras model
    
    '''
    # initialize model
    model = Sequential() 

    model.add(layers.Dense(size1))
    model.add(layers.Dense(size2))
    model.add(layers.Dense(size3))
    # can extend to more layers if needed!
 
    model.add(layers.Dropout(dropout))

    # must return len of class size
    model.add(layers.Dense(output_len))
    # pass final output through activation function so that it is a probability vector
    model.add(layers.Activation('softmax'))

    # compile model using an optimizer and loss function
    optimizer = Adam(learning_rate=learning_rate, clipnorm=1)
    model.compile(optimizer=optimizer, loss='categorical_crossentropy')

    return model

def train(config, train_x_ds, train_y_ds, val_x_ds, val_y_ds, model_name, model_path, show_perf=True):
    '''Trains the neural network using the parameters specified in the config file. Saves the model to the models directory.
    
    Params:
        config (dict): dictionary of parameters to be used in training the neural network
        train_x_ds (numpy array): numpy array of the input training data
        train_y_ds (numpy array): numpy array of the ouput training targets
        val_x_ds (numpy array): numpy array of the input validation data
        val_y_ds (numpy array): numpy array of the ouput validation targets
        model_name (str): name of the model to be saved
        model_path (str): path to the models directory
        data_dir (str): path to the data directory
        show_perf (bool): whether to show the training history plot or not

    Returns:
        None, but saves the model to the models directory
    
    '''
    # make sure you don't already have a model with this name
    assert not os.path.exists(os.path.join(model_path, model_name+'.keras')), f'Model {model_name} already exists!'

    # get length of output vector
    output_len = train_y_ds.shape[1]

    # initialize the neural net
    model = build_model(config['size_1'],  config['size_2'], config['size_3'], 
            config['dropout'], config['learning_rate'], output_len)
    
    #now run training
    history = model.fit(train_x_ds, train_y_ds, epochs=config['epochs'], batch_size=config['batch_size'], validation_data=(val_x_ds, val_y_ds))
    
    # save model as h5 and config for later use
    model.save(os.path.join(model_path, model_name+'.keras'))
    with open(os.path.join(model_path, model_name+'.txt'), 'w') as f:
        f.write(str(config))

    # now plot training history to see performance over time
    if show_perf:
        plt.figure(figsize=(10, 10))
        plt.plot(history.history['loss'], label='Train')
        plt.plot(history.history['val_loss'], label='Val')
        plt.title(f'Loss for {model_name}')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.savefig(os.path.join(model_path, f'training_history_{model_name}.pdf'))
        plt.show()

if __name__ == '__main__':
    # define config dictionary
    # this is really the hardest part of the ML process: choosing the right hyperparameters! for now I encourage you to play around with the following values and see how the performance changes to get a sense for what each parameter does
    # NOTE: you can also use the wandb sweep feature to do a hyperparameter search for you! I've included an example of how to do this in the commented out section at the bottom of this file
    config = {
        'size_1': 256,
        'size_2': 180,
        'size_3': 56,
        'dropout': 0.2,
        'learning_rate': 0.001,
        'epochs': 5, # number of times to run through the training data
        'batch_size': 64 # number of training examples to use in each training step
    }

    # define path to where your data and models are
    DATA_DIR = '/Users/oscarscholin/Desktop/Pomona/Senior_Year/Fall2024/Astro_proj/UCVS/data'
    MODEL_DIR = '/Users/oscarscholin/Desktop/Pomona/Senior_Year/Fall2024/Astro_proj/UCVS/supervised/fall2023/models'

    # load datasets
    train_x_ds = np.load(os.path.join(DATA_DIR, 'train_x_ds.npy'), allow_pickle=True)
    train_y_ds = np.load(os.path.join(DATA_DIR, 'train_y_ds.npy'), allow_pickle=True)
    val_x_ds = np.load(os.path.join(DATA_DIR, 'val_x_ds.npy'), allow_pickle=True)
    val_y_ds = np.load(os.path.join(DATA_DIR, 'val_y_ds.npy'), allow_pickle=True)

    # print the data
    # print('first input of train_x', train_x_ds[1])
    # print('first output of train_y', train_y_ds[1])

    # train the model
    model_name = 'oscar_model1'
    train(config, train_x_ds, train_y_ds, val_x_ds, val_y_ds, model_name, MODEL_DIR, DATA_DIR)

## ----------- WANDB code ----------- ##
# WandB (wandb.ai) is a super useful tool for hyperparameter optimization by performing "sweeps", which are basically different configurations drawn from distributions of your choice. You can do this manually by iterating through values in nested for loops and feeding this config file manually like we do above, but this can get cumbersome. Here's how we can modify the code in this file to use wandb sweeps instead:

# import wandb

# add the following lines as the first line in the def train() function: 
    # with wandb.init(config=config):
        # config = wandb.config



#set nested dictionary with wanb
# set dictionary with random search; optimizing val_loss--------------------------
# sweep_config= {
#     'method': 'random',
#     'name': 'val_accuracy',
#     'goal': 'maximize'
# }

# sweep_config['metric']= 'val_accuracy'

# parameters_dict = {
#     'epochs': {
#        'distribution': 'int_uniform',
#        'min': 20,
#        'max': 100
#     },
#     # for build_dataset
#      'batch_size': {
#        'values': [x for x in range(32, 161, 32)]
#     },
#     'size_1': {
#        'distribution': 'int_uniform',
#        'min': 64,
#        'max': 256
#     },
#     'size_2': {
#        'distribution': 'int_uniform',
#        'min': 64,
#        'max': 256
#     },
#     'size_3': {
#        'distribution': 'int_uniform',
#        'min': 64,
#        'max': 256
#     },
#     'learning_rate':{
#          #uniform distribution between 0 and 1
#          'distribution': 'uniform', 
#          'min': 0,
#          'max': 0.1
#      },
#      'dropout':{
#             #uniform distribution between 0 and 1
#             'distribution': 'uniform', 
#             'min': 0,
#             'max': 0.5
#         }
# }

# # append parameters to sweep config
# sweep_config['parameters'] = parameters_dict 

# # login to wandb 
# wandb.init(project="[enter proj name here]", entity="your_username")

# # initialize sweep agent
# sweep_id = wandb.sweep(sweep_config, project='[enter proj name here]', entity="oscarscholin")
# wandb.agent(sweep_id, train, count=100) # count is the number of different configurations to try