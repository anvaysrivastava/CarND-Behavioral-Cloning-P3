#Import the data
import csv
import cv2
import numpy as np
import random
from itertools import chain
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Conv2D
from keras.layers.pooling import MaxPooling2D
from keras.layers.core import Dropout


lines = []

center_images = []
measurements = []
model_name = 'model.h5'
data_path1 = "data"
data_path2 = "manual_data"
data_path3 = "smooth_driving"
data_path4 = "dirt_curve1"
correction = 0.5


def readCsv(path):
	print("Starting the import of data from ",path)
	with open(path + '/driving_log.csv') as csv_file:
		reader = csv.reader(csv_file)
		for line in reader:
			lines.append(line)

def getCenterImage():
	for line in lines:
		if ".jpg" in line[0]:
			yield cv2.imread(line[0].strip())

def getCenterSteering():
	for line in lines:
		if ".jpg" in line[0]:
			yield float(line[3])

def getFlippedImage():
	for line in lines:
		if ".jpg" in line[0]:
			yield np.fliplr(cv2.imread(line[0].strip()))

def getFlippedSteering():
	for line in lines:
		if ".jpg" in line[0]:
			yield -float(line[3])

def getLeftImage():
	for line in lines:
		if ".jpg" in line[0]:
			yield cv2.imread(line[1].strip())

def getLeftSteering():
	for line in lines:
		if ".jpg" in line[0]:
			yield float(line[3]) + correction

def getRightImage():
	for line in lines:
		if ".jpg" in line[0]:
			yield cv2.imread(line[2].strip())

def getRightSteering():
	for line in lines:
		if ".jpg" in line[0]:
			yield float(line[3]) - correction

readCsv(data_path1)
readCsv(data_path2)
readCsv(data_path3)
readCsv(data_path4)


X_train = np.array(list(chain(getCenterImage(), getFlippedImage(), getLeftImage(), getRightImage())))
Y_train = np.array(list(chain(getCenterSteering(), getFlippedSteering(), getLeftSteering(), getRightSteering())))
print(X_train[0].shape)	#160x320x3
print("X size ", X_train.size)
print("Y size ", Y_train.size)
print("Import of data complete")

#Train the data

print("Trainig Started")

model = Sequential()
model.add(Lambda(lambda x:x-128.0, input_shape=(X_train[0].shape)))
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(X_train[0].shape))) #65x320x3
model.add(Conv2D(5, (3, 3), padding='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(8, (5, 5), padding='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5, 5), padding='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dropout(rate=0.1))
model.add(Dense(200))
model.add(Dense(100))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, validation_split=0.2, shuffle=True, epochs=9)
print(model_name)
model.save(model_name)
