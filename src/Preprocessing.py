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
from PIL import Image
import os

#these can be adjusted for your files as needed
#is file path my filepath is awful hopefully yours will be better
#i have no idea what format your images will be in so ive compiled a list of those that I can think of if yours isnt there tough add it in
        
for files in os.listdir('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder/real_and_fake_face_detection/real_and_fake_face/training_fake'):
    if files.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff')): 
        imagePath=os.path.join('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder/real_and_fake_face_detection/real_and_fake_face/training_fake',files)
        try:
            img=Image.open(imagePath)
            #a new filepath to save the images to
            newFilepath=os.path.join('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder/GreyScaledImages',files)
            #processing face extraction and greyscale for now

            #converts to greyscale using L rather than P L is black and white whereas P is color palate
            greyImage=img.convert('L')
            greyImage.save(newFilepath)
            
            img.close()
        except Exception as e:
            print(f"Error in processing {files}:{e}")