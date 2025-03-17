'''
    This file handles the work of getting an image ready to be processed from
    the model for determination of Real vs. Fake. The image is first turned into
    a grey-scaled version; then be put through two of the pre-processing methods
    that remove the Background from the Foreground. To ensure time processing the
    image through two different methods are kept relatively low, multi-processing
    is used to run each method on their own separate thread. Once both methods have
    completed their functions, then the main thread will take over and run a
    comparison method to get the likeness of each image. This results in the final
    pre-processed image ready for actual neural processing.
'''

# Imports
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
from Util import Action, Constants
from PIL import Image

class Preprocess():
    '''
        Class to handle the preprocessing methods and actions
    '''

    def __init__(self):
        '''
            Constructor for the Preprocess class
        '''
        pass

    def perform(self) -> bool:
        '''
            This performs the operation of pre-processing the data into the corresponding location and information.
            First, the folder locations are checked to determine what course of action to pursue. If the files exist, then no\n
            pre-processing is necessary. If not, then extraction and saving of the data is performed. Two different methods are\n
            invoked when handling the images, as such Multi-Processing is used to perform in parallel the methods of FG/BG segergaion.\n
            Finally, the images are then compared and combined with only their similarities intact to use as the processing image.
        '''
        pass

    def __ensureFolders(self):
        '''Ensures that the folders for model exist.'''
        fldr: list[str] = ['zip', 'extracted', 'grey']
        for f_name in fldr:
            path: str = '\\'.join([Constants.DIR_TRAINING, f_name])

            if(not os.path.exists(path)):
                os.mkdir(path)