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
        
        pass


    def __check_files(self) -> bool:
        '''
            This checks if there are files within the extracted files directory. If not, then this will\n
            check if the training folder contains any ZIP files. Should ZIP files exist without any extracted\n
            files, then this will open each archive and save the data into their respective folders.
        '''
        return True

        
# for files in os.listdir('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder/real_and_fake_face_detection/real_and_fake_face/training_fake'):
#     if files.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')): 
#         imagePath=os.path.join('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder/real_and_fake_face_detection/real_and_fake_face/training_fake',files)
#         try:
#             img=Image.open(imagePath)
#             #a new filepath to save the images to
#             newFilepath=os.path.join('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder/GreyScaledImages',files)
#             #processing face extraction and greyscale for now

#             #converts to greyscale using L rather than P L is black and white whereas P is color palate
#             greyImage=img.convert('L')
#             greyImage.save(newFilepath)
            
#             img.close()
#         except Exception as e:
#             print(f"Error in processing {files}:{e}")