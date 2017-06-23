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
import matplotlib.image as mpimg


lines = []

model_name = 'model.h5'
data_path1 = "data"
data_path2 = "manual_data"
data_path3 = "smooth_driving"
data_path4 = "dirt_curve1"
data_path5 = "dirt_curve2"
correction = 0.5


def readCsv(path):
	print("Starting the import of data from ",path)
	with open(path + '/driving_log.csv') as csv_file:
		reader = csv.reader(csv_file)
		for line in reader:
			lines.append(line)

def getData(batch_size):
	images = np.empty([0, 160, 320, 3], dtype=float)
	measurements = np.empty([0, ], dtype=float)
	count = 0
	for line in lines:
		count += 1
		if ".jpg" in line[0]:
			if count<batch_size:
				center_image = mpimg.imread(line[0].strip())
				center_steering = float(line[3])
				center_fliped_image = np.fliplr(center_image)

				left_image = mpimg.imread(line[1].strip())
				left_steering = float(line[3]) + correction
				left_fliped_image = np.fliplr(left_image)

				right_image = mpimg.imread(line[2].strip())
				right_steering = float(line[3]) - correction
				right_fliped_image = np.fliplr(right_image)


				images = np.append(images,np.array([center_image, center_fliped_image, left_image, left_fliped_image, right_image, right_fliped_image]), axis = 0)
				measurements = np.append(measurements,np.array([center_steering, -center_steering, left_steering, -left_steering, right_steering, -right_steering]),axis = 0)
			else:
				yield images, measurements
	yield images, measurements

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

def getFlippedLeftImage():
	for line in lines:
		if ".jpg" in line[0]:
			yield np.fliplr(cv2.imread(line[1].strip()))

def getFlippedLeftSteering():
	for line in lines:
		if ".jpg" in line[0]:
			yield -float(line[3]) - correction

def getFlippedRightImage():
	for line in lines:
		if ".jpg" in line[0]:
			yield np.fliplr(cv2.imread(line[2].strip()))

def getFlippedRightSteering():
	for line in lines:
		if ".jpg" in line[0]:
			yield -float(line[3]) + correction



readCsv(data_path1)
readCsv(data_path2)
readCsv(data_path3)
readCsv(data_path4)
readCsv(data_path5)

total_lines = len(lines)
epochs = 6


X_train = np.array(list(chain(getCenterImage(), getFlippedImage(), getLeftImage(), getRightImage(), getFlippedLeftImage(), getFlippedRightImage())))
Y_train = np.array(list(chain(getCenterSteering(), getFlippedSteering(), getLeftSteering(), getRightSteering(), getFlippedLeftSteering(), getFlippedRightSteering())))
print("X size ", X_train.shape)
print("Y size ", Y_train.shape)
print("Import of data complete")

#Train the data

print("Trainig Started")

model = Sequential()
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160, 320, 3))) #65x320x3
model.add(Lambda(lambda x:x/32.0-4.0))
model.add(Conv2D(1, (1, 1), padding='valid', activation='elu'))
# model.add(MaxPooling2D())
model.add(Conv2D(3, (3, 3), padding='valid', activation='elu'))
model.add(MaxPooling2D())
# model.add(MaxPooling2D())
# model.add(Dropout(rate=0.1))
model.add(Conv2D(3, (3, 3), padding='valid', activation='elu'))
model.add(MaxPooling2D())

model.add(Conv2D(6, (5, 5), padding='valid', activation='elu'))
model.add(MaxPooling2D())
# model.add(MaxPooling2D())
# model.add(Dropout(rate=0.1))
model.add(Conv2D(16, (5, 5), padding='valid', activation='elu'))
# model.add(MaxPooling2D())
# model.add(MaxPooling2D())
# model.add(MaxPooling2D())
# model.add(Dropout(rate=0.1))
model.add(Flatten())
# model.add(Dropout(rate=0.1))
model.add(Dense(250))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(1))
# model.add(Dropout(rate=0.1))
model.compile(loss='mse', optimizer='adam')

# model.fit_generator(getData(10), steps_per_epoch=total_lines, epochs = epochs, initial_epoch = 0)
model.fit(X_train, Y_train, validation_split=0.05, shuffle=True, epochs=2)
print(model_name)
model.save(model_name)
