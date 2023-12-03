import pandas as pd
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop
from openpyxl import Workbook
import openpyxl


# Read the data from the CSV file
data = pd.read_csv('data_pion_20.csv')

# Split the data into input and output variables
#X = data.drop('sqrt', axis=1) #static input for each case : extra data of fiting
#X = data.drop('massno', axis=1) #static input for each case : extra data of fiting
#X = data.drop('output', axis=1)
X = data['y']
y = data['output']

# Define the model
model = Sequential()
#model.add(Dense(7, input_dim=X.shape[1], activation='relu'))
model.add(Dense(7, input_dim=1, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with Levenberg-Marquardt optimizer
optimizer = RMSprop(lr=0.001, rho=0.001, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X, y, epochs=72, batch_size=16, validation_split=0.2)

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
output = pd.DataFrame({'y': X.values.flatten(), 'Actual': y.values.flatten()})
outputpredicat = pd.DataFrame({'y': X.values.flatten(), 'Actual': y.values.flatten(), 'Predicted': predictions.flatten()})
# output is data frame

# Write the DataFrames to an Excel file with three sheets
with pd.ExcelWriter('predict_Pion_20.xlsx') as writer:
    outputpredicat.to_excel(writer, sheet_name='predicat', index=False)
    output.to_excel(writer, sheet_name='output', index=False)
    


#output.to_csv('predict_Pion_20_as_paper.csv', index=False)

print("End")