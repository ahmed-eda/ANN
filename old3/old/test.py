import pandas as pd
from keras.models import Sequential
from keras.layers import Dense

# Read the data from the CSV file
data = pd.read_csv('data.csv')

# Split the data into input and output variables
X = data.drop('output', axis=1)
y = data['output']

input_test=X.shape[1]
print(" input_test")
print(input_test)
print("stat network")
# Define the model
model = Sequential()
model.add(Dense(7, input_dim=X.shape[1], activation='relu'))
model.add(Dense(5, activation='relu'))
#model.add(Dense(16, activation='relu'))
model.add(Dense(1, activation='linear'))
#model.add(Dense(1, activation='sigmoid'))

# Compile the model
#model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.compile(loss='mean_squared_error', optimizer='adam')

# Train the model
model.fit(X, y, epochs=72, batch_size=32, validation_split=0.2)

# Make predictions on new data
new_data = pd.read_csv('data_20.csv').drop('output', axis=1)
print("new_data is : ")
print(new_data)
predictions = model.predict(new_data)
print("predictions is : ")
print(predictions)
print("End")