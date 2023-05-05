# first neural network with keras make predictions
from numpy import loadtxt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import csv


# load the dataset
#dataset = loadtxt('data.csv', delimiter=',',converters=float)
file = open("data.csv", "r")
dataset = list(csv.reader(file, delimiter=","))
file.close()
print(dataset)
# split into input (X) and output (y) variables
X = dataset[:,0:3]
y = dataset[:,3]

print(x)
print(y)
# define the keras model
model = Sequential()
model.add(Dense(8, input_shape=(4,), activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
# compile the keras model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# fit the keras model on the dataset
model.fit(X, y, epochs=150, batch_size=4, verbose=0)
# make class predictions with the model
predictions = (model.predict(X) > 0.5).astype(int)
# summarize the first 5 cases
for i in range(5):
	print('%s => %d (expected %d)' % (X[i].tolist(), predictions[i], y[i]))