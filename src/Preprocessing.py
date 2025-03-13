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
from PIL import Image
from Util import Constants, Action

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
            This performs the operation of pre-processing the data into the corresponding location and information.\n
            First, the folder locations are checked to determine what course of action to pursue. If the files exist, then no\n
            pre-processing is necessary. If not, then extraction and saving of the data is performed. Two different methods are\n
            invoked when handling the images, as such Multi-Processing is used to perform in parallel the methods of FG/BG segergaion.\n
            Finally, the images are then compared and combined with only their similarities intact to use as the processing image.
        '''

        if(self.__check_files()):
            for folder in os.listdir(os.path.join(Constants.DIR_TRAINING, "extractedFiles")):
                if(not os.path.isdir(folder)):
                    continue

                for files in os.listdir(folder):
                    if(files.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff'))):
                        try:
                            img = Image.open(os.path.join(folder, files))
                            newImPath = os.path.join(Constants.DIR_TRAINING, "gray" + "/temp/" + files)

                            grey = img.convert('L')
                            grey.save(newImPath)
                        except Exception as e:
                            print(f"Error in processing {files}:{e}")
        
        for folder in os.listdir(os.path.join(Constants.DIR_TRAINING, "extractedFiles")): # TODO: Need to implement multi-processing
            self.__performWatershed(folder)
            self.__performContour(folder)

        print(f'Pre-Processing Complete.')

    def __performWatershed(path: str):
        '''
            This separates the Foreground from the background using the Watershed method.
        '''
        for files in os.listdir(path):
            image = cv2.imread(files)
            ret, threst = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

            # Remove noise
            kernel = np.ones((3,3), np.uint8)
            opening = cv2.morphologyEx(threst, cv2.MORPH_OPEN, kernel, iterations = 2)

            # Ensure correct background
            ensureBG = cv2.dilate(open, kernel, iterations = 3)
            
            # Ensure correct foreground
            distTransform  = cv2.distanceTransform(opening, cv2.DIST_L2, 5)
            ret, ensureFG = cv2.threshold(distTransform, 0.7 * distTransform.max(), 255, 0)

            # find unknown regions
            ensureFG = np.uint8(ensureFG)
            unknown = cv2.subtract(ensureBG, ensureFG)
            
            # Marker labeling
            ret, markers = cv2.connectedComponents(ensureFG)
            markers += 1
            markers[unknown == 255] = 0

            # Perform watershed method
            markers = cv2.watershed(image, markers)
            image[markers == -1] = [255, 0, 0]
    
    def __performContour(path: str):
        '''
            This draws the contour lines the separate the Foreground from the background.
        '''
        for files in os.listdir(path):
            # Input image. (Note: This image should be GREY-SCALED).
            image = cv2.imread(files)

            # Thresh image and find the Contours of the image
            ret, thresh = cv2.threshold(image, 127, 255, 0)
            contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            # Draw the Contours onto the image for further processing
            for contour in contours:
                cv2.drawContours(image, [contour], 0, (0,255,0), 3)

    def __check_files(self) -> bool:
        '''
            This checks if there are files within the extracted files directory. If not, then this will\n
            check if the training folder contains any ZIP files. Should ZIP files exist without any extracted\n
            files, then this will open each archive and save the data into their respective folders.
        '''
        return True