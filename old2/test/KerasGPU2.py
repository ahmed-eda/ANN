

import keras
#from keras.utils.training_utils import multi_gpu_model
#from keras.utils.traceback_utils import multi_gpu_model
import keras.utils.traceback_utils as kk
import keras.utils.traceback_utils 
import keras
#from keras.utils import multi_gpu_model
import pandas as pd


# input variable to program
inputFile = './datasets/data 2017_pi+ 7.7.xlsx'
inputSheetName = 'main'
outputFile = 'out_in3.xlsx'
outputSheetName = 'predicat_in3'
nameFigImg = 'fig_in3.png'

print(inputFile)

# Read the data from the CSV file
data = pd.read_excel(inputFile,sheet_name=inputSheetName)
print(data)
#data = data[data['N part']==337]


X = data.drop('Spectrum', axis=1)
#X = data['Pt']
y = data['Spectrum']

x_train = X
y_train = y

# Create a model
model = keras.models.Sequential([
    keras.layers.Dense(12, input_dim=3, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])


import tensorflow as tf
from tensorflow import keras

# Check if GPU is available
physical_devices = tf.config.list_physical_devices('GPU')
print("Num GPUs:", len(physical_devices))

# Create a multi-GPU model
import keras
from keras.utils.tf_utils import multi_gpu_model


multi_gpu_model = multi_gpu_model(model, gpus=2)

# Train the model
multi_gpu_model.fit(x_train, y_train, epochs=10, batch_size=32)