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
import zipfile as zip
from Util import Action, Constants
from PIL import Image

def perform(force: bool = False) -> bool:
    '''
        This performs the operation of pre-processing the data into the corresponding location and information.
        First, the folder locations are checked to determine what course of action to pursue. If the files exist, then no\n
        pre-processing is necessary. If not, then extraction and saving of the data is performed. Two different methods are\n
        invoked when handling the images, as such Multi-Processing is used to perform in parallel the methods of FG/BG segergaion.\n
        Finally, the images are then compared and combined with only their similarities intact to use as the processing image.
    '''

    # Ensure all folders for the data exist
    # Extract all files from any Zip files
    # Convert extracted into greyscale images.
    __ensureFolders()
    __extractFiles(force)
    __greyScale(Action.toPath(Constants.DIR_TRAINING, 'extracted'), [Constants.DIR_TRAINING, 'grey'], force)

def __extractFiles(force: bool = False) -> bool:
    for zips in os.listdir(Action.toPath(Constants.DIR_TRAINING, 'zip')):
        if(not os.path.exists(Action.toPath(Constants.DIR_TRAINING, 'extracted', zips)) or force):
            if(os.path.exists(Action.toPath(Constants.DIR_TRAINING, 'extracted', zips))):
                os.remove(Action.toPath(Constants.DIR_TRAINING, 'extracted'))

            print(f'Extracting files from {zips}...')
            zFile = zip.ZipFile(Action.toPath(Constants.DIR_TRAINING, 'zip', zips))
            zFile.extractall(Action.toPath(Constants.DIR_TRAINING, 'extracted'))
            print(f'Complete')
    return True

def __greyScale(path: str, deep: list[str] = [], force: bool = False) -> None:
    '''
        This will convert all images found within the `extracted` folder into their greyscale version.
        The method is recursive to handle inner (deep) folders. The naming of the folders and files will
        remain the same when saving from the `extracted` to the `grey` directory.

        Args:
            path (str, required): This is the path of the current directory
            deep (list[str], optional): List that contains the lower folder paths
            force (bool, optional): Can force the re-doing of the greyscaling of an image, even if the file already exists.
    '''
    
    for item in os.listdir(path):
        tmpPath = Action.toPath(path, item)
        
        if os.path.isdir(tmpPath):
            deep.append(item)
            __greyScale(tmpPath, deep, force)
            deep.pop()

        elif item.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')):
            if(os.path.exists(Action.toPath(deep, item)) and not force):
                continue

            imgPath = tmpPath

            #TODO: Need to add the ability to resize an image to 224x224. This can be done utilizing PIL

            try:
                img = Image.open(imgPath)
                savePath = Action.toPath(deep, item)

                #converts to greyscale using L rather than P L is black and white whereas P is color palate
                grey = img.convert('L')

                if(not os.path.exists(Action.toPath(deep))):
                    os.makedirs(Action.toPath(deep))

                grey.save(savePath)
            
                img.close()
            except IOError as e:
                print(f'Error, could not handle file {e.filename}. Exception: {e}')

        else:
            continue

def __ensureFolders() -> None:
    '''Ensures that the folders for model exist.'''

    fldr: list[str] = ['zip', 'extracted', 'grey']
    for f_name in fldr:
        path: str = '\\'.join([Constants.DIR_TRAINING, f_name])

        if(not os.path.exists(path)):
            os.makedirs(path)