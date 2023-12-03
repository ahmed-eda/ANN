import keras
from keras.utils import multi_gpu_model
import pandas as pd


# input variable to program
inputFile = '../datasets/data 2017_pi+ 7.7.xlsx'
inputSheetName = 'main'
outputFile = 'out_in3.xlsx'
outputSheetName = 'predicat_in3'
nameFigImg = 'fig_in3.png'

# Read the data from the CSV file
data = pd.read_excel(inputFile,sheet_name=inputSheetName)
#data = data[data['N part']==337]
# Split the data into input and output variables
#X = data.drop('sqrt', axis=1) #static input for each case : extra data of fiting
#X = data.drop('massno', axis=1) #static input for each case : extra data of fiting
#X = data.drop('output', axis=1)

X = data.drop('Spectrum', axis=1)
#X = data['Pt']
y = data['Spectrum']

x_train = X
y_train = y

# Create a model.
model = keras.Sequential([
    keras.layers.Dense(12, input_dim=3, activation='relu'),
    keras.layers.Dense(8, activation='relu'),
    keras.layers.Dense(4, activation='relu'),
    keras.layers.Dense(1, activation='sigmoid')
])

# Wrap the model in a multi_gpu_model object.
parallel_model = multi_gpu_model(model, gpus=2)

# Compile the model.
parallel_model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])


# Train the model.
parallel_model.fit(x_train, y_train, epochs=10)
