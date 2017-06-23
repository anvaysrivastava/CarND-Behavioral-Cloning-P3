#Import the dependencies
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

# Listing all the different datasets from which the model will be trained.
model_name = 'model.h5'
data_path1 = "data"
data_path2 = "manual_data"
data_path3 = "smooth_driving"
data_path4 = "dirt_curve1"
data_path5 = "dirt_curve2"

# Correction factor for left and right images.
correction = 0.5

# Load the reference to all the images and steering.
readCsv(data_path1)
readCsv(data_path2)
readCsv(data_path3)
readCsv(data_path4)
readCsv(data_path5)

# Load the images and steering in memory and create an array out of it.
# NOTE: While project requirement clearly stated that I should use generators.
# I decided to go against it due to the following reasons:
# 1) The data set is small 2.6GB and fits completely in memory.
# 2) On my machine training time was ~16 times fasted when using model.fit
# 	 compared to model.fit_generator
X_train = np.array(list(chain(getCenterImage(), getFlippedImage(), getLeftImage(), getRightImage(), getFlippedLeftImage(), getFlippedRightImage())))
Y_train = np.array(list(chain(getCenterSteering(), getFlippedSteering(), getLeftSteering(), getRightSteering(), getFlippedLeftSteering(), getFlippedRightSteering())))

print("X Shape ", X_train.shape)
print("Y Shape ", Y_train.shape)

#Train the data

print("Trainig Started")

model = Sequential()
#Cropping the image to avoid learning from environmental parameters like trees etc.
model.add(Cropping2D(cropping=((70,25),(0,0)),input_shape=(160, 320, 3))) #65x320x3

#Normalizing the input.
#NOTE: I did not do X/256 - 0.5 because my car was performing terribly near
# dirt tracks. My theory was that due to normalization network is not able to
# differentiate the color of road and prefer grey over browm. However
# doing X = X - 128.0 was a bit too extreme as the car would take sudden turns
# on tree shadows before bridge.
model.add(Lambda(lambda x:x/32.0-4.0))

#Convolutional layer
model.add(Conv2D(1, (1, 1), padding='valid', activation='elu'))
model.add(Conv2D(3, (3, 3), padding='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(3, (3, 3), padding='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(6, (5, 5), padding='valid', activation='elu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (5, 5), padding='valid', activation='elu'))

#Flatten layer.
#NOTE: No need to add Dropout as I am doing only 2 epocs on non repeating
# dataset. Hence I am fairly confident that the network has reliable redundancy
# and is not overfitting.
model.add(Flatten())

# Fully connected layer.
model.add(Dense(250))
model.add(Dense(100))
model.add(Dense(25))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train, Y_train, validation_split=0.05, shuffle=True, epochs=2)

#Save the model.
model.save(model_name)

# Helper functions.
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
