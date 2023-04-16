import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from openpyxl import Workbook
import openpyxl
from keras.models import load_model
import numpy as np


# Read the data from the CSV file
data = pd.read_csv('data_pion_20.csv')

# Split the data into input and output variables
#X = data.drop('sqrt', axis=1) #static input for each case : extra data of fiting
#X = data.drop('massno', axis=1) #static input for each case : extra data of fiting
#X = data.drop('output', axis=1)
X = data['y']
y = data['output']

# Define the model
""" model = Sequential()
model.add(Dense(7, input_dim=1, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(28, activation='relu'))
model.add(Dense(14, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(1, activation='linear')) """
#model.add(Dense(7, input_dim=X.shape[1], activation='relu'))
#model.add(Dense(7, input_dim=1, activation='relu'))
#model.add(Dense(5, activation='sigmoid'))
#model.add(Dense(1, activation='relu'))

# Compile the model with Levenberg-Marquardt optimizer
""" optimizer = RMSprop(lr=0.001, rho=0.001)
model.compile(loss='mean_squared_error', optimizer=optimizer)
#model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=500, batch_size=10, validation_split=0.2)


# Save the model
model.save('pion20_500.h5') """



# Load the saved model
model = load_model('pion20_1000.h5')



# Make predictions on new data
#new_data = pd.read_csv('data_pion_20.csv').drop('output', axis=1)
new_data = pd.read_csv('data_pion_20.csv')['y']
#predictions = model.predict(new_data)
print("new_data is : ")
print(new_data)
predictions = model.predict(new_data)
print("predictions is : ")
print(predictions)


# Plot the data and predictions
plt.plot(X, y, 'bo', label='Actual')
plt.plot(X, predictions, 'r', label='Predicted')
plt.xlabel('X')
plt.ylabel('Output')
plt.legend()
plt.show()

# Write predictions and plot data to Excel file
#output = pd.DataFrame({'y': X.values.flatten(), 'Actual': y.values.flatten()})
outputpredicat = pd.DataFrame({'y': X.values.flatten(), 'Actual': y.values.flatten(), 'Predicted': predictions.flatten()})
# output is data frame

# Write the DataFrames to an Excel file with three sheets
with pd.ExcelWriter('predict_Pion_20_modified.xlsx') as writer:
    outputpredicat.to_excel(writer, sheet_name='predicat', index=False)
    #output.to_excel(writer, sheet_name='output', index=False)
    


#output.to_csv('predict_Pion_20_as_paper.csv', index=False)



# print some information about the model
print(model.summary())

# Evaluate the model on the training data
result = model.evaluate(X, y)

# Check if the result is a tuple or a single float value
if isinstance(result, tuple):
    mse, mae = result
else:
    mse = result
    mae = None

# Print the MSE and MAE
print('MSE:', mse)
if mae is not None:
    print('MAE:', mae)


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


# Get the weights and biases of the model
#weights, biases = model.get_weights()

# Extract the weight and bias for the first layer
#w = weights[0][0]
#b = biases[0]

W = model.get_weights()

print("W : ")
print("W : ")
print(W)
print("W : ")
print("W type : ")

print(type(W))
print("W shap : ")
# Loop through each layer in the model
for layer in model.layers:
    # Check if the layer is a Dense layer
    if isinstance(layer, Dense):
        # Get the weights for the layer
        weights = layer.get_weights()[0]
        # Print the shape of the weights
        print('Layer:', layer.name)
        print('Weights shape:', weights.shape)


# Print the fitting equation
#print('y =', w, '* x +', b)

print("End")
print("End")