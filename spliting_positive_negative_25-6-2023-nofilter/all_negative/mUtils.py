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
    excel_files = [
        f for f in os.listdir(".") if f.endswith(".xlsx") and not f.startswith("out")
    ]
    if not excel_files:
        raise ValueError("No Excel files found in the current directory")
    return excel_files[0]


class MyParam:
    inputFile = None
    inputSheetName = None
    data = None
    X = None
    y = None

def init_param(p):
    p.inputFile = get_input_file_name()
    p.inputSheetName = "Sheet1"
    p.data = None
    p.X = None
    p.y = None
    return p


def get_data_in_Param(p):
    # Read the data from the excel file in class of parameters
    
    data_all = pd.read_excel(p.inputFile, sheet_name=p.inputSheetName)
    temp_data_all = data_all  # data_all[data_all['spectrum']<60]
    data = temp_data_all.reset_index(drop=True)
    # Split the data into input and output variables
    X = data[["mass", "s", "N part", "Pt"]]
    y = data["spectrum"].to_frame("spectrum")
    p.data = data
    p.X = X
    p.y = y
    print(p.X.head)
    print(p.y.head)
    return p


""" def InitParam():
    # return class of Param
    # call the parameters
    myInitParam = MyParam()

    # return it
    return myInitParam
 """