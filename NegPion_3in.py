import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from openpyxl import Workbook
import openpyxl
from keras.models import load_model

# input variable to program
inputFile = 'datasets/data 2017_pi+ 7.7.xlsx'
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



# Load the saved model
model = load_model('NegPion_L9.h5')

# Define the model
'''
model = Sequential()
model.add(Dense(3, input_dim=3, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(27, activation='relu'))



model.add(Dense(27, activation='relu'))
model.add(Dense(9, activation='relu'))
model.add(Dense(3, activation='relu'))
model.add(Dense(1, activation='linear'))
'''

# Compile the model with Levenberg-Marquardt optimizer
#optimizer = RMSprop(lr=0.001, rho=0.001)
#model.compile(loss='mean_squared_error', optimizer=optimizer)
#model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
#model.fit(X, y, epochs=100, batch_size=32, validation_split=0.2)
#model.fit(X, y, epochs=1000, batch_size=32)


# Save the model
#model.save('NegPion_L9.h5')

# Load the saved model
#model = load_model('pion20_1000.h5')

# Make predictions on new data
#new_data = pd.read_csv('data_pion_20.csv').drop('output', axis=1)
new_data = X
#predictions = model.predict(new_data)
#new_data = new_data.drop('Spectrum', axis=1)

#newX = new_data['Pt']
newX = new_data

print("new_data is : ")
print(new_data)
predictions = model.predict(new_data)
print("predictions is : ")
print(predictions)

# for drawing in 2d i choose Pt as x-axis
xg = new_data['Pt']


# Plot the data and predictions
plt.plot(xg, y, 'bo', label='Actual')
plt.plot(xg, predictions, 'ro', label='Predicted')
plt.xlabel('newX')
plt.ylabel('Output')
plt.legend()

plt.semilogy(xg,  y)
plt.semilogy(xg, predictions)

plt.scatter(xg, y)
plt.scatter(xg, predictions)


plt.show()
plt.savefig(nameFigImg)

# Write predictions and plot data to Excel file
#output = pd.DataFrame({'y': X.values.flatten(), 'Actual': y.values.flatten()})
#outputpredicat = pd.DataFrame({'y': X.values.flatten(), 'Actual': y.values.flatten(), 'Predicted': predictions.flatten()})
outputpredicat = pd.DataFrame({ 'Pt': xg.values.flatten(),'Actual': y.values.flatten(), 'Predicted': predictions.flatten()})
# output is data frame

# Write the DataFrames to an Excel file with three sheets
with pd.ExcelWriter(outputFile) as writer:
    outputpredicat.to_excel(writer, sheet_name=outputSheetName, index=False)
    #output.to_excel(writer, sheet_name='output', index=False)
    


print("End")
print(model.summary())



