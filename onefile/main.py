# Lib
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

# define the parameter
test = 'test'
# data parameters
outFolder = 'out'
inputFolder = 'data'
modelNamePath = ''
inputFile = ''
inputSheetName = ''
data = ''
X = ''
y = ''
X_train = ''  
X_train_part = ''
X_test = ''
X_test_part = ''
y_test_part = ''
y_train_part = ''
predictions = ''
datap = pd.DataFrame(data=None)
mergedData = ''
outputpredicat = ''
# Model parameteres  
model = None
myepochs = int(0)
mybatchSize = int(0)
modelName = ''
score = 0
mse = 0
rmse = 0
# Output parameters
outputFile = ''
summaryOutFile = ''
outputSheetName = ''
nameFigImg = ''
nameAllFigImg = ''

# Utility function

# Get current data excel file name (doesn't start with out* ) 
def get_input_file_name(self):
    readFolder = self.inputFolder
    print('os.listdir("readFolder") : ',os.listdir(readFolder))
    excel_files = [
        f for f in os.listdir(readFolder) if f.endswith(".xlsx") and not f.startswith("out")
    ]
    if not excel_files:
        raise ValueError("No Excel files found in the current directory")
    return excel_files[0]
# get model file name
def get_model_fromfile(self):
    _modelNames = [f for f in os.listdir('.') if (f.endswith('.h5'))]        
    print(' get model from file _modelNames[0] : ',_modelNames[0] )
    if not _modelNames:
        raise ValueError("No model files found in function the current directory")
    return _modelNames[0]






# define the main function
def main():
# Start the main function
    print("Starting main call")

    print(" type the test :",test)
    
# End the main function
    print("The End of main ")






# call the main function
if __name__ == "__main__":
    main()
