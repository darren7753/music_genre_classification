import json
import logging
import os
import subprocess
import shutil
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sliceSpecs import slice_spec

from sklearn.metrics import confusion_matrix
import itertools

import torch


#define a Dictionary that maps song genre to labels

label_to_genre = {0: "Electronic", 1: "Experimental", 2: "Folk", 3: "Hip-Hop", 4: "Instrumental", 5: "International", 6: "Pop", 7: "Rock"}

class Params():
    """Class that loads hyperparameters from a json file.

    Example:
    ```
    params = Params(json_path)
    print(params.learning_rate)
    params.learning_rate = 0.5  # change the value of learning_rate in params
    ```
    """

    def __init__(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)
        if 'cuda' not in self.__dict__:
            self.cuda = False

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)
            
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class RunningAverage():
    """A simple class that maintains the running average of a quantity
    
    Example:
    ```
    loss_avg = RunningAverage()
    loss_avg.update(2)
    loss_avg.update(4)
    loss_avg() = 3
    ```
    """
    def __init__(self):
        self.steps = 0
        self.total = 0
    
    def update(self, val):
        self.total += val
        self.steps += 1
    
    def __call__(self):
        return self.total/float(self.steps)
        
    
def set_logger(log_path):
    """Set the logger to log info in terminal and file `log_path`.

    In general, it is useful to have a logger so that every output to the terminal is saved
    in a permanent file. Here we save it to `model_dir/train.log`.

    Example:
    ```
    logging.info("Starting training...")
    ```

    Args:
        log_path: (string) where to log
    """
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if not logger.handlers:
        # Logging to a file
        file_handler = logging.FileHandler(log_path)
        file_handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logger.addHandler(file_handler)

        # Logging to console
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(stream_handler)


def save_dict_to_json(d, json_path):
    """Saves dict of floats in json file

    Args:
        d: (dict) of float-castable values (np.float, int, float, etc.)
        json_path: (string) path to json file
    """
    with open(json_path, 'w') as f:
        # We need to convert the values to float for json (it doesn't accept np.array, np.float, )
        d = {k: float(v) for k, v in d.items()}
        json.dump(d, f, indent=4)


def save_checkpoint(state, is_best, checkpoint):
    """Saves model and training parameters at checkpoint + 'last.pth.tar'. If is_best==True, also saves
    checkpoint + 'best.pth.tar'

    Args:
        state: (dict) contains model's state_dict, may contain other keys such as epoch, optimizer state_dict
        is_best: (bool) True if it is the best model seen till now
        checkpoint: (string) folder where parameters are to be saved
    """
    filepath = os.path.join(checkpoint, 'last.pth.tar')
    if not os.path.exists(checkpoint):
        print("Checkpoint Directory does not exist! Making directory {}".format(checkpoint))
        os.mkdir(checkpoint)
    else:
        print("Checkpoint Directory exists! ")
    torch.save(state, filepath)
    if is_best:
        shutil.copyfile(filepath, os.path.join(checkpoint, 'best.pth.tar'))


def load_checkpoint(checkpoint, model, optimizer=None):
    """Loads model parameters (state_dict) from file_path. If optimizer is provided, loads state_dict of
    optimizer assuming it is present in checkpoint.

    Args:
        checkpoint: (string) filename which needs to be loaded
        model: (torch.nn.Module) model for which the parameters are loaded
        optimizer: (torch.optim) optional: resume optimizer from checkpoint
    """
    if not os.path.exists(checkpoint):
        raise("File doesn't exist {}".format(checkpoint))
    checkpoint = torch.load(checkpoint)
    model.load_state_dict(checkpoint['state_dict'])

    if optimizer:
        optimizer.load_state_dict(checkpoint['optim_dict'])

    return checkpoint

def plot_confusion_matrix_2(cm, classes,
                          normalize=False,
                          save_plot = False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    if save_plot:
        plt.savefig("conf_matrix_plot.svg")
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    
    
def plot_confusion_matrix(y_actu, y_pred, save_plot, title='Confusion matrix', cmap=plt.cm.gray_r):
    y_actu = y_actu.reshape((y_actu.shape[1], ))
    y_pred = y_pred.reshape((y_pred.shape[1], ))
    
    cnf_matrix = confusion_matrix(y_actu, y_pred);
    
    np.set_printoptions(precision=2)
    
    class_names = np.array(['Elect.', 'Expertl', 'Folk', 'Hip-Hop', 'Instrl', 'Intnl', 'Pop', 'Rock'])
    
    plt.figure()
    plot_confusion_matrix_2(cnf_matrix, classes=class_names, save_plot = save_plot, normalize = True)
    plt.show()
    
#     y_actu = y_actu.reshape((y_actu.shape[1], ))
#     y_pred = y_pred.reshape((y_pred.shape[1], ))
    
#     df_confusion = pd.crosstab(y_actu, y_pred, rownames=['Actual'], colnames=['Predicted'], margins=True)
    
#     df_conf_norm = df_confusion / df_confusion.sum(axis=1)
#     print(df_confusion)
    
#     plt.matshow(df_confusion, cmap=cmap) # imshow
#     #plt.title(title)
#     plt.colorbar()
#     tick_marks = np.arange(len(df_confusion.columns))
#     plt.xticks(tick_marks, df_confusion.columns, rotation=45)
#     plt.yticks(tick_marks, df_confusion.index)
#     if save_plot:
#         plt.tight_layout()
#         plt.savefig("conf_matrix_plot.svg")
        
#     plt.ylabel(df_confusion.index.name)
#     plt.xlabel(df_confusion.columns.name)
#     plt.show()


    
    
def plot_bar_graph(genre_confidences):
    """
    Plots a horizontal bar graph given a vector of confidences of each song genre

    Uses the the label_to_genre dictionary defined above to convert vector indices to bar labels

    arguments:
    genre_confidences := numpy array with values between 0 and 1 representing the track genre confidences
    """
    x = [values for _, values in label_to_genre.items()]
    df = pd.DataFrame(genre_confidences)
    ax = df.plot.barh()
    ax.set_yticklabels(x)
    plt.tight_layout()
    plt.savefig("bar_plot.svg")
    plt.show()
    


def preprocess_track_for_classification(path_to_song, output_dir):
    """
    Preprocesses a song before classifying it into a genre.

    The steps include:
      - convert song to .wav format if mp3
      - Create the song's spectrogram and save it in a temporary folder
      - Slice the created spectrograms and save them into data/128x128_specs/tmp_specs/

    """

    # Clear all temporary data folder first; This ensures you don't mix spectrograms
    # different songs

#     output_dir = "data/128x128_specs/tmp_specs"
    tmp_test_folder = "test_songs/"

    if os.path.exists(output_dir):
        c = ["rm", "-rf", output_dir]
        subprocess.call(c)

    if not os.path.exists(tmp_test_folder):
        os.mkdir(tmp_test_folder)
    else:
        print("Warning: temporary directory {} already exists".format(tmp_test_folder))

        
    if not os.path.exists(path_to_song):
        print("The path {} does not exist".format(path_to_song))
        quit()
    else:
        fname = path_to_song
        
    # Convert to .wav first
    # Note, this makes it work on a Mac since I couldn't get Sox to handle .mp3. However, on the Ubuntu machines
    # this is unnecessary
    if fname.endswith(".mp3"):
        dst = os.path.join(tmp_test_folder, fname.split("/")[-1].split(".mp")[0] + ".wav")
        command = ['ffmpeg', '-v', '0', '-i', fname, dst]
        subprocess.call(command)

    fname = os.path.join(tmp_test_folder, fname.split("/")[-1].split(".")[0] + ".wav")
        
    # Create a spectrogram using sox
    dest = fname.split(".wa")[0] + ".png"
    command = ['sox', fname, '-n', 'remix', '1','spectrogram', '-Y', '200', '-X', '50', '-m', '-r', '-o', dest]
    subprocess.call(command)

    #slice and save spectrograms
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)
    new_dest= dest.split("/")[-1]
    slice_spec(new_dest, 128, tmp_test_folder, output_dir)
    
    
    

    
