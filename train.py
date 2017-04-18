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

correction = 0.1
for line in lines:
	if ".jpg" in line[0]:
		# Taking center image first
		image = cv2.imread(data_path + line[0].strip())
		steering = float(line[3])
		if steering != 0:
			center_images.append(image)
			measurements.append(steering)
			center_images.append(np.fliplr(image))
			measurements.append(-steering)

		# Taking left image
		left_image = cv2.imread(data_path + line[1].strip())
		left_steering = float(line[3]) + correction
		center_images.append(left_image)
		measurements.append(left_steering)

		# Taking right image
		image = cv2.imread(data_path + line[2].strip())
		steering = float(line[3]) - correction
		center_images.append(image)
		measurements.append(steering)
X_train = np.array(center_images)
Y_train = np.array(measurements)
print(X_train[0].shape)
print("Import of data complete")

#Train the data

print("Trainig Started")
from keras.models import Sequential
from keras.layers import Flatten, Dense, Lambda, Cropping2D, Convolution2D, Dropout

model = Sequential()
model.add(Lambda(lambda x:x/255.0-0.5, input_shape=(X_train[0].shape)))
model.add(Cropping2D(cropping=((70,25),(0,0))))	#image = 320x65x3
model.add(Convolution2D(3, 1, 1, border_mode='same')) # 320x65x3
model.add(Convolution2D(6, 5, 1, border_mode='valid', activation='elu')) # 64x13x6
model.add(Convolution2D(25, 3, 1, border_mode='valid', activation='elu')) # 31x4x25
model.add(Convolution2D(20, 2, 1, border_mode='valid', activation='elu')) # 15x2x10
model.add(Flatten()) # 200
# model.add(Dropout(0.01))
model.add(Dense(100))
model.add(Dense(50))
model.add(Dense(10))
model.add(Dense(1))
model.compile(loss='mse', optimizer='adam')

model.fit(X_train, Y_train, validation_split=0.20, shuffle=True, nb_epoch=3)

model.save('model.h5')
