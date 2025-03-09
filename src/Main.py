'''
    Inports
'''
import os
import datetime
from Util import Constants

if(__name__ == "__main__"):
    print(os.path.join(Constants.CURRENT_DIR, datetime.datetime.now().strftime('%Y-%m-%d_%H:%M:%S')))
