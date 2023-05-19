import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from openpyxl import Workbook
import openpyxl
from keras.models import load_model



def GetWeightsBiases():
    # print some information about the model
    print(model.summary())
    # Loop through each layer in the model
    for layer in model.layers:
        # Check if the layer is a Dense layer
        if isinstance(layer, Dense):
            # Get the weights and biases for the layer
            weights, biases = layer.get_weights()
            # Print the weights and biases
            print('Layer:', layer.name)
            print('Weights:', weights)
            print('Biases:', biases)