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
    inputFile = get_input_file_name()
    inputSheetName = "Sheet1"
    data = None
    X = None
    y = None


def get_data_in_Param():
    # Read the data from the excel file in class of parameters
    p = MyParam()
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


def InitParam():
    # return class of Param
    # call the parameters
    myInitParam = MyParam()
    # return it
    return myInitParam


# start the main function
def main():
    print("Starting main call")
    p = InitParam()
    p = get_data_in_Param()
    print("p.y : ", p.y)

    # Then we will start at normalizing data
    print("The End")


if __name__ == "__main__":
    main()
