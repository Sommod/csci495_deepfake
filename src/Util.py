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

    def unzip_files(self, zipPath: str, fileLoc: str = Constants.DIR_CURRENT) -> bool:
        '''
        Name: unzip_files\n
        Type: Function\n
        Description: Unzips an archive. By default, the Timestamp and current directory are used
                    as the extraction location, but a location can be given using the fileLoc
                    parameter.\n
        Parameters:\n
            @param zipPath - Location of the zip archive
            @param fileLoc - Location of extraction\n
        Return:\n
            True - If files were successfully extracted, otherwise False\n
        '''
        
        # Try-Catch for extracting files from Zip File
        try:
            files = zip.ZipFile(zipPath, "r")

            if(fileLoc == Constants.CURRENT_DIR):
                fileLoc = os.path.join(fileLoc, self.get_timestamp(), "/")

            files.extractall(fileLoc)
            files.close()
            return True
        except:
            print(f'Error: Could not extract the files of {zipPath}.')
            return False
        
    def get_timestamp():
        '''
            Gets the current time, useful for file/folder naming
        '''
        return datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')