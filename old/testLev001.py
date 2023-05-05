import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import RMSprop

# Read the data from the CSV file
data = pd.read_csv('data.csv')

# Split the data into input and output variables
X = data.drop('output', axis=1)
y = data['output']

# Define the model
model = Sequential()
model.add(Dense(7, input_dim=X.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='linear'))

# Compile the model with Levenberg-Marquardt optimizer
optimizer = RMSprop(lr=0.01, rho=0.9, epsilon=None, decay=0.0)
model.compile(loss='mean_squared_error', optimizer=optimizer)

# Train the model
model.fit(X, y, epochs=72, batch_size=32, validation_split=0.2)

# Make predictions on new data
new_data = pd.read_csv('data_20.csv').drop('output', axis=1)
#predictions = model.predict(new_data)
print("new_data is : ")
print(new_data)
predictions = model.predict(new_data)
print("predictions is : ")
print(predictions)

# Write predictions to CSV file
output = pd.DataFrame(predictions, columns=['predictions'])
output.to_csv('predictions.csv', index=False)

print("End")