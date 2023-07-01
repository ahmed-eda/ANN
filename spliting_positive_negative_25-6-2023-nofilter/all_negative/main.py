
# function to call all needed headers lib
import pandas as pd
import matplotlib.pyplot as plt    
import numpy as np
import os
from keras.models import Sequential
from keras.layers import Dense, BatchNormalization
from keras.optimizers import RMSprop
from keras.models import load_model
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_squared_error
from io import StringIO



def get_input_file_name():
    excel_files = [f for f in os.listdir('.') if f.endswith('.xlsx') and not f.startswith('out')]
    if not excel_files:
        raise ValueError("No Excel files found in the current directory")
    return excel_files[0]

class MyParam:
    inputFile = get_input_file_name()
    inputSheetName = 'Sheet1'

def InitParam():
    # return class of Param
    # call the parameters 
    myInitParam = MyParam()
    # return it
    return myInitParam

# start the main function 
def main():
    print('Starting main call')
    p = InitParam()
    
    
    print('The End')

    
if __name__ == "__main__":
    main()
