"""
Authors: Josh Moore,

Description: This file is used as a Utility purpose. This provides quick re-usable code for cleaner code usage for 
any manipulation.

Log:
    19 Jan 2025:
	- Created Util file
    09 March 2025:
    - Added handler for Zip Files and Extraction
    - Added Constant Class
"""

# Imports
import zipfile as zip
import os
import datetime

class Constants:
    '''
    Type: Class\n
    Name: Constants\n
    Description: This class is used specifically to house constant values that can be
                 used without the need to re-initialize or calculate.
    '''

    def __init__(self):
        pass

    DIR_CURRENT = os.path.dirname(os.path.realpath(__file__))
    '''Current Directory of the Program'''

    DIR_TRAINING = os.path.join(DIR_CURRENT, "trainingDataDirectory")
    '''Top-Level Directory for the Training / Extracted files'''

    DIR_OUTPUT = os.path.join(DIR_CURRENT, "output")
    '''Top-Level Directory for the Output files and data'''


#import your data as a zip file this cell will extract your images into the DeepfakeDetectionImagesFolder 
#the next cells will process the images as needed
#theoretically we will all put our zip files in here and extract them all into the folder then rearrange the folder so that all images are in thier own grouped files
### files =zip.ZipFile("DeepfakeDetectionFiles/real_and_fake_face_detection.zip",'r' )#change real_and_fake_face as needed for your zip file name
### files.extractall('DeepfakeDetectionFiles/DeepfakeDetectionImageFolder')
### files.close()
class Action:
    def __init__(self):
        pass
        
    def extract(self, force: bool = False) -> bool:
        '''Extracts the Zip files into the extracted folder

        This finds all the .ZIP files within the 'zip' folder and extracts each one
        into their unzipped format within the 'extracted' folder. The name of the
        zip file is also the name used for the extracted folder name. When checking
        if the extracted data already exists, the names of the ZIP files are used.
        Additionally, the function can be forced to override already extracted data.

        Args:
            force (bool, optional): Forces the function to extract the files, overidding existing folders.

        Returns:
            bool: `True` If zip files were successfully extracted, `False` otherwise
        '''
        try:
            zipPath: list[str] = [Constants.DIR_TRAINING, 'zip']
            extractPath: list[str] = [Constants.DIR_TRAINING, 'extracted']

            for zips in os.listdir('\\'.join(zipPath)):
                files = zip.ZipFile(os.path.join('\\'.join(zipPath), zips), 'r')

                files.extractall('\\'.join(extractPath))
                files.close()
            return True
        
        except IOError as e:
            print(f'Error, could not extract {files} without error.')
            return False
        
    def get_timestamp() -> str:
        '''
            Gets the current time, useful for file/folder naming

            Returns:
                str: String format of the DateTime
        '''
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    
    def toPath(**items) -> str:
        ret: str = ''

        for i in items.items():
            ret = os.path.join(ret, i)

        return ret