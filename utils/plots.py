import os
import pickle
from datetime import datetime

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set() # matplotlib plots are more beautiful with seaborn ッ

VISUALIZATION_DIR = 'visualizations'

def check_dir_exist():
    if not os.path.exists(VISUALIZATION_DIR):
        os.mkdir(VISUALIZATION_DIR)

def plot_accuracy(train_accuracy: list, valid_accuracy: list, title: str = 'Training Accuracy'):
    """This function plots the training and validation accuracy."""
    
    assert len(train_accuracy) == len(valid_accuracy), f'Number of train accuracy ({len(train_accuracy)}) \
                                                    is not equal to validation accuracy ({len(valid_accuracy)})'
    
    # Plot
    plt.figure()
    plt.plot(train_accuracy, label='train')
    plt.plot(valid_accuracy, label='valid')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title(title)
    plt.legend()
    
    # Save plot
    check_dir_exist()
    filename = title + '_' + str(datetime.now())
    plt.savefig(os.path.join(VISUALIZATION_DIR, filename+'.png'))
    data = {'train_accuracy':train_accuracy, 'valid_accuracy':valid_accuracy}
    pickle.dump(data, open(os.path.join(VISUALIZATION_DIR,filename+'.pkl'), 'wb'))