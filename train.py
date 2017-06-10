#Import the data
data_path = "data/"
print("Starting the import of data")
import csv
import cv2
import numpy as np
import random

lines = []

with open(data_path + 'driving_log.csv') as csv_file:
	reader = csv.reader(csv_file)
	for line in reader:
		lines.append(line)

center_images = []
measurements = []
model_name = 'model.h5'

correction = 0.1
for line in lines:
	if ".jpg" in line[0]:
		image = cv2.imread(data_path + line[0].strip())
		steering = float(line[3])
		if steering!=0:
			center_images.append(image)
			measurements.append(steering)
			center_images.append(np.fliplr(image))
			measurements.append(-steering)
		else:
			center_images.append(np.fliplr(image))
			measurements.append(-steering)
			center_images.append(image)
			measurements.append(steering)
		left_image = cv2.imread(data_path + line[1].strip())
		left_steering = float(line[3]) + correction
		center_images.append(left_image)
		measurements.append(left_steering)
		image = cv2.imread(data_path + line[2].strip())
		steering = float(line[3]) - correction
		center_images.append(image)
		measurements.append(steering)
X_train = np.array(center_images)
Y_train = np.array(measurements)
print(X_train[0].shape)	#160x320x3
print("Import of data complete")

#Train the data

print("Trainig Started")
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(X_train[0].shape)))
model.add(Cropping2D(cropping=((70,25),(0,0)))) #65x320x3
model.add(Convolution2D(1, 1, 1, border_mode='valid', subsample=(1, 10), activation='elu'))
model.add(Convolution2D(3, 3, 3, border_mode='valid', activation='elu'))
model.add(Convolution2D(6, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))
model.add(Convolution2D(16, 5, 5, border_mode='valid', subsample=(2, 2), activation='elu'))
model.add(Flatten())
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, nb_epoch=2)
print(model_name)
model.save(model_name)
