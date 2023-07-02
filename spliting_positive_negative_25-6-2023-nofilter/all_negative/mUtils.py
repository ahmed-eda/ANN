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


class MyParam:
    def __init__(self):
        # data parameters
        self.inputFile = ''
        self.inputSheetName = ''
        self.data = ''
        self.X = ''
        self.y = ''
        self.X_train = ''  
        # Model parameteres  
        self.myepochs = 0
        self.mybatchSize = 0
        self.modelName = ''
        # Output parameters
        self.outputFile = ''
        self.summaryOutFile = ''
        self.outputSheetName = ''
        self.nameFigImg = ''
        pass

# utility function

    # Get current data excel file name (doesn't start with out* ) 
    def get_input_file_name():
        excel_files = [
            f for f in os.listdir(".") if f.endswith(".xlsx") and not f.startswith("out")
        ]
        if not excel_files:
            raise ValueError("No Excel files found in the current directory")
        return excel_files[0]

    # get model file name
    def get_model_fromfile():
        modelNames = [f for f in os.listdir('.') if (f.endswith('.h5'))]
        if not modelNames:
            raise ValueError("No Excel files found in the current directory")
        return modelNames[0]

# Methods

# init
    def init_param(self,_inputfile='',sheetName='Sheet1'):
        # data parameters
        if(len(_inputfile)>0):
            self.inputFile = _inputfile
        else:
            self.inputFile = self.get_input_file_name() # get current excel file in the folder
        
        self.inputSheetName = "Sheet1"
        self.data = ''
        self.X = ''
        self.y = ''
        self.X_train = ''  
        # Model parameteres  
        self.myepochs = int(100) 
        self.mybatchSize = int(16)
        self.modelName = ''
        # Output parameters
        self.outputFile = ''
        self.summaryOutFile = ''
        self.outputSheetName = ''
        self.nameFigImg = ''
    
# read data
    def get_data_in_Param(self):
        # Read the data from the excel file in class of parameters
        data_all = pd.read_excel(p.inputFile, sheet_name=p.inputSheetName)
        temp_data_all = data_all  # data_all[data_all['spectrum']<60]
        data = temp_data_all.reset_index(drop=True)
        # Split the data into input and output variables
        self.X = data[["mass", "s", "N part", "Pt"]]
        self.y = data["spectrum"].to_frame("spectrum")
        self.data = data
        print(self.X.head)
        print(self.y.head)
        
# normaliz input
    def normaliz_data(self):
        # Normalize the input
        from sklearn.preprocessing import RobustScaler
        # Create a RobustScaler object
        scaler = RobustScaler()
        # Fit the scaler to the input data and transform it
        X_normalized = scaler.fit_transform(self.X)
        # Print the normalized input data
        print('X_normalized')
        print(X_normalized)
        self.X_train = X_normalized
        
 # set model and config param
    def set_model_config_param(self,_modelName=''):
        if(len(_modelName)>0):
            self.modelName=_modelName
        else:
            self.modelName = self.get_model_fromfile()
        
        if len(self.modelName)>0:
            raise ValueError("No model files found in the current directory")
        
        # Output parameters
        self.outputFile = 'out_ '+self.modelName+' .xlsx'
        self.summaryOutFile = self.modelName + ' _ Summary .txt'
        self.outputSheetName = 'predicat_ '+self.modelName+' '
        self.nameFigImg = 'fig_in4_ '+self.modelName+' .png'

 # end       
    

