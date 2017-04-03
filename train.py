#Import the data
data_path = "data/"
print("Starting the import of data")
import csv
import cv2
import numpy as np

lines = []

with open(data_path + 'driving_log.csv') as csv_file:
	reader = csv.reader(csv_file)
	for line in reader:
		lines.append(line)

center_images = []
measurements = []

for line in lines:
	if ".jpg" in line[0]:	
		image = cv2.imread(data_path + line[0])
		center_images.append(image)
		measurements.append(float(line[3]))
		center_images.append(np.fliplr(image))
		measurements.append(-float(line[3]))
X_train = np.array(center_images)
Y_train = np.array(measurements)
print(X_train[0].shape)
print("Import of data complete")

#Train the data

print("Trainig Started")
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()
model.add(Flatten(input_shape=X_train[0].shape))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=7)

model.save('model.h5')
